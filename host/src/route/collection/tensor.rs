use std::convert::TryInto;

use futures::future::{self, Future, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt, TryStreamExt};
use log::debug;
use safecast::*;

use tc_btree::Node;
use tc_error::*;
use tc_tensor::*;
use tc_transact::fs::{CopyFrom, Dir};
use tc_transact::Transaction;
use tc_value::{
    Bound, Number, NumberClass, NumberInstance, NumberType, Range, TCString, Value, ValueType,
};
use tcgeneric::{label, Label, PathSegment, TCBoxTryFuture, Tuple};

use crate::collection::{Collection, DenseTensor, DenseTensorFile, SparseTensor, Tensor};
use crate::fs;
use crate::route::{AttributeHandler, GetHandler, PostHandler, PutHandler, SelfHandlerOwned};
use crate::scalar::Scalar;
use crate::state::State;
use crate::stream::TCStream;
use crate::txn::Txn;

use super::{Handler, Route};

const AXIS: Label = label("axis");
const TENSORS: Label = label("tensors");

struct CastHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for CastHandler<T>
where
    T: TensorTransform + Send + Sync + 'a,
    Tensor: From<T::Cast>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let dtype =
                    ValueType::try_cast_from(key, |v| TCError::bad_request("not a NumberType", v))?;

                let dtype = dtype.try_into()?;
                self.tensor
                    .cast_into(dtype)
                    .map(Tensor::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for CastHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct ConcatenateHandler;

impl ConcatenateHandler {
    async fn blank(
        txn: &Txn,
        shape: Vec<u64>,
        dtype: NumberType,
    ) -> TCResult<DenseTensor<DenseTensorFile>> {
        let txn_id = *txn.id();
        let file = txn
            .context()
            .create_file_unique(txn_id, TensorType::Dense)
            .await?;

        DenseTensor::constant(file, txn_id, shape, dtype.zero()).await
    }

    async fn concatenate(
        txn: &Txn,
        shape_in: Shape,
        dtype: NumberType,
        tensors: Vec<Tensor>,
    ) -> TCResult<Tensor> {
        let mut shape_out = Vec::with_capacity(shape_in.len() + 1);
        shape_out.push(tensors.len() as u64);
        shape_out.extend(shape_in.iter().cloned());

        let concatenated = Self::blank(txn, shape_out, dtype).await?;

        let mut writes: FuturesUnordered<_> = tensors
            .into_iter()
            .enumerate()
            .map(|(i, tensor)| {
                let bounds = vec![AxisBounds::At(i as u64)].into();
                concatenated.clone().write(txn.clone(), bounds, tensor)
            })
            .collect();

        while let Some(()) = writes.try_next().await? {
            // no-op
        }

        Ok(concatenated.into())
    }

    async fn concatenate_axis(
        txn: &Txn,
        shape_in: Shape,
        axis: usize,
        dtype: NumberType,
        tensors: Vec<Tensor>,
    ) -> TCResult<Tensor> {
        let mut shape_out = shape_in.to_vec();
        shape_out[axis] = shape_in[axis] * tensors.len() as u64;

        let bounds: Bounds = shape_in.iter().map(|dim| AxisBounds::all(*dim)).collect();

        let concatenated = Self::blank(txn, shape_out, dtype).await?;

        let mut writes: FuturesUnordered<_> = tensors
            .into_iter()
            .enumerate()
            .map(|(i, tensor)| {
                let start = i as u64 * shape_in[axis];
                let mut bounds = bounds.clone();
                bounds[axis] = AxisBounds::In(start..(start + shape_in[axis]));
                concatenated.clone().write(txn.clone(), bounds, tensor)
            })
            .collect();

        while let Some(()) = writes.try_next().await? {
            // no-op
        }

        Ok(concatenated.into())
    }
}

impl<'a> Handler<'a> for ConcatenateHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let tensors: Vec<Tensor> = params.require(&TENSORS.into())?;
                let axis: Value = params.or_default(&AXIS.into())?;
                params.expect_empty()?;

                if tensors.is_empty() {
                    return Err(TCError::unsupported(
                        "need at least one Tensor to concatenate",
                    ));
                }

                let shape_in = tensors[0].shape().clone();
                for i in 0..tensors.len() {
                    if tensors[i].shape() != &shape_in {
                        return Err(TCError::unsupported(
                            "can only concatenate Tensors with the same shape",
                        ));
                    }
                }

                let dtype = tensors
                    .iter()
                    .map(TensorAccess::dtype)
                    .fold(tensors[0].dtype(), Ord::max);

                if axis.is_none() {
                    Self::concatenate(txn, shape_in, dtype, tensors).await
                } else {
                    let axis = cast_axis(axis, shape_in.len())?;
                    Self::concatenate_axis(txn, shape_in, axis, dtype, tensors).await
                }
                .map(Collection::Tensor)
                .map(State::Collection)
            })
        }))
    }
}

struct ConstantHandler;

impl<'a> Handler<'a> for ConstantHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let (shape, value): (Vec<u64>, Number) =
                    key.try_cast_into(|v| TCError::bad_request("invalid Tensor schema", v))?;

                constant(&txn, shape.into(), value)
                    .map_ok(Tensor::from)
                    .map_ok(Collection::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

struct CopyFromHandler;

impl<'a> Handler<'a> for CopyFromHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let source = params.require(&label("tensor").into())?;
                params.expect_empty()?;

                let copy = match source {
                    Tensor::Dense(source) => {
                        let file = txn
                            .context()
                            .create_file_unique(*txn.id(), TensorType::Dense)
                            .await?;

                        let blocks =
                            BlockListFile::copy_from(source.into_inner(), file, txn).await?;

                        DenseTensor::from(blocks.accessor()).into()
                    }
                    Tensor::Sparse(source) => {
                        let dir = txn.context().create_dir_unique(*txn.id()).await?;
                        let table = SparseTable::copy_from(source, dir, txn).await?;
                        SparseTensor::from(table.accessor()).into()
                    }
                };

                Ok(State::Collection(Collection::Tensor(copy)))
            })
        }))
    }
}

struct CopyDenseHandler;

impl<'a> Handler<'a> for CopyDenseHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let Schema { dtype, shape } =
                    schema.try_cast_into(|v| TCError::bad_request("invalid Tensor schema", v))?;

                let source: TCStream = params.require(&label("source").into())?;
                params.expect_empty()?;

                let elements = source.into_stream(txn.clone()).await?;
                let elements = elements.map(|r| {
                    r.and_then(|n| {
                        Number::try_cast_from(n, |n| {
                            TCError::bad_request("invalid Tensor element", n)
                        })
                    })
                });

                let txn_id = *txn.id();
                let file = create_file(txn).await?;
                DenseTensorFile::from_values(file, txn_id, shape, dtype, elements)
                    .map_ok(DenseTensor::from)
                    .map_ok(Tensor::from)
                    .map_ok(Collection::Tensor)
                    .map_ok(State::Collection)
                    .await
            })
        }))
    }
}

struct CopySparseHandler;

impl<'a> Handler<'a> for CopySparseHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let schema: Schema =
                    schema.try_cast_into(|v| TCError::bad_request("invalid Tensor schema", v))?;

                let source: TCStream = params.require(&label("source").into())?;
                params.expect_empty()?;

                let elements = source.into_stream(txn.clone()).await?;

                let txn_id = *txn.id();
                let dir = txn.context().create_dir_unique(txn_id).await?;
                let tensor = SparseTensor::create(&dir, schema, txn_id).await?;

                let elements = elements
                    .map(|r| {
                        r.and_then(|state| {
                            Value::try_cast_from(state, |s| {
                                TCError::bad_request("invalid sparse Tensor element", s)
                            })
                        })
                    })
                    .map(|r| {
                        r.and_then(|row| {
                            row.try_cast_into(|v| {
                                TCError::bad_request(
                                    "sparse Tensor expected a (Coord, Number) tuple, found",
                                    v,
                                )
                            })
                        })
                    });

                elements
                    .map_ok(|(coord, value)| tensor.write_value_at(txn_id, coord, value))
                    .try_buffer_unordered(num_cpus::get())
                    .try_fold((), |(), ()| future::ready(Ok(())))
                    .await?;

                Ok(Collection::Tensor(tensor.into()).into())
            })
        }))
    }
}

struct CreateHandler {
    class: TensorType,
}

impl<'a> Handler<'a> for CreateHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let schema: Schema =
                    key.try_cast_into(|v| TCError::bad_request("invalid Tensor schema", v))?;

                match self.class {
                    TensorType::Dense => {
                        constant(&txn, schema.shape, schema.dtype.zero())
                            .map_ok(Tensor::from)
                            .map_ok(Collection::Tensor)
                            .map_ok(State::Collection)
                            .await
                    }
                    TensorType::Sparse => {
                        let txn_id = *txn.id();
                        let dir = txn.context().create_dir_unique(txn_id).await?;

                        SparseTensor::create(&dir, schema, txn_id)
                            .map_ok(Tensor::from)
                            .map_ok(Collection::Tensor)
                            .map_ok(State::Collection)
                            .await
                    }
                }
            })
        }))
    }
}

struct EinsumHandler;

impl<'a> Handler<'a> for EinsumHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let format: TCString = params.require(&label("format").into())?;
                let tensors: Vec<Tensor> = params.require(&TENSORS.into())?;
                einsum(&format, tensors)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

struct ElementsHandler<T> {
    tensor: T,
}

impl<T> ElementsHandler<T> {
    fn new(tensor: T) -> Self {
        Self { tensor }
    }
}

impl<'a, T> Handler<'a> for ElementsHandler<T>
where
    T: TensorAccess + TensorTransform + Send + Sync + 'a,
    Tensor: From<T> + From<T::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let tensor = if key.is_none() {
                    Tensor::from(self.tensor)
                } else {
                    let bounds = cast_bounds(self.tensor.shape(), key)?;
                    let slice = self.tensor.slice(bounds)?;
                    Tensor::from(slice)
                };

                Ok(TCStream::from(Collection::Tensor(tensor)).into())
            })
        }))
    }
}

struct ExpandHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for ExpandHandler<T>
where
    T: TensorAccess + TensorTransform + Send + 'a,
    Tensor: From<T::Expand>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let axis = cast_axis(key, self.tensor.ndim())?;

                self.tensor
                    .expand_dims(axis)
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for ExpandHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct FlipHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for FlipHandler<T>
where
    T: TensorAccess + TensorTransform + Send + 'a,
    Tensor: From<T::Flip>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let axis = cast_axis(key, self.tensor.ndim())?;
                self.tensor
                    .flip(axis)
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for FlipHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct RangeHandler;

impl<'a> Handler<'a> for RangeHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<(Vec<u64>, Number, Number)>() {
                    let (shape, start, stop): (Vec<u64>, Number, Number) =
                        key.opt_cast_into().unwrap();

                    let file = create_file(&txn).await?;

                    DenseTensor::range(file, *txn.id(), shape, start, stop)
                        .map_ok(Tensor::from)
                        .map_ok(Collection::from)
                        .map_ok(State::from)
                        .await
                } else {
                    Err(TCError::bad_request("invalid schema for range tensor", key))
                }
            })
        }))
    }
}

struct TransposeHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for TransposeHandler<T>
where
    T: TensorTransform + Send + 'a,
    Tensor: From<T::Transpose>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let transpose = if key.is_none() {
                    self.tensor.transpose(None)
                } else {
                    let permutation = key.try_cast_into(|v| {
                        TCError::bad_request("invalid permutation for transpose", v)
                    })?;

                    self.tensor.transpose(Some(permutation))
                };

                transpose
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for TransposeHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

impl Route for TensorType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(CreateHandler { class: *self }));
        } else if path.len() != 1 {
            return None;
        }

        match self {
            Self::Dense => match path[0].as_str() {
                "copy_from" => Some(Box::new(CopyDenseHandler)),
                "concatenate" => Some(Box::new(ConcatenateHandler)),
                "constant" => Some(Box::new(ConstantHandler)),
                "range" => Some(Box::new(RangeHandler)),
                _ => None,
            },
            Self::Sparse => match path[0].as_str() {
                "copy_from" => Some(Box::new(CopySparseHandler)),
                _ => None,
            },
        }
    }
}

struct DualHandler {
    tensor: Tensor,
    op: fn(Tensor, Tensor) -> TCResult<Tensor>,
    op_const: fn(Tensor, Number) -> TCResult<Tensor>,
}

impl DualHandler {
    fn new<T>(
        tensor: T,
        op: fn(Tensor, Tensor) -> TCResult<Tensor>,
        op_const: fn(Tensor, Number) -> TCResult<Tensor>,
    ) -> Self
    where
        Tensor: From<T>,
    {
        Self {
            tensor: tensor.into(),
            op,
            op_const,
        }
    }
}

impl<'a> Handler<'a> for DualHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, r| {
            Box::pin(async move {
                let r = Number::try_cast_from(r, |r| {
                    TCError::bad_request("expected a Number, not", r)
                })?;

                (self.op_const)(self.tensor, r)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let l = self.tensor;
                let r = params.remove(&label("r").into()).ok_or_else(|| {
                    TCError::bad_request("missing right-hand-side parameter r", &params)
                })?;

                params.expect_empty()?;

                match r {
                    State::Collection(Collection::Tensor(r)) => {
                        if l.shape() == r.shape() {
                            (self.op)(l, r).map(Collection::from).map(State::from)
                        } else {
                            let (l, r) = broadcast(l, r)?;
                            (self.op)(l, r).map(Collection::from).map(State::from)
                        }
                    }
                    State::Scalar(Scalar::Value(r)) if r.matches::<Number>() => {
                        let r = r.opt_cast_into().unwrap();
                        (self.op_const)(l, r).map(Collection::from).map(State::from)
                    }
                    other => Err(TCError::bad_request(
                        "expected a Tensor or Number, found",
                        other,
                    )),
                }
            })
        }))
    }
}

struct ReduceHandler<'a, T: TensorReduce<fs::Dir>> {
    tensor: &'a T,
    reduce: fn(T, usize) -> TCResult<<T as TensorReduce<fs::Dir>>::Reduce>,
    reduce_all: fn(&'a T, Txn) -> TCBoxTryFuture<'a, Number>,
}

impl<'a, T: TensorReduce<fs::Dir>> ReduceHandler<'a, T> {
    fn new(
        tensor: &'a T,
        reduce: fn(T, usize) -> TCResult<<T as TensorReduce<fs::Dir>>::Reduce>,
        reduce_all: fn(&'a T, Txn) -> TCBoxTryFuture<'a, Number>,
    ) -> Self {
        Self {
            tensor,
            reduce,
            reduce_all,
        }
    }
}

impl<'a, T> Handler<'a> for ReduceHandler<'a, T>
where
    T: TensorAccess + TensorReduce<fs::Dir> + Clone + Sync,
    Tensor: From<<T as TensorReduce<fs::Dir>>::Reduce>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    (self.reduce_all)(self.tensor, txn.clone())
                        .map_ok(Value::from)
                        .map_ok(State::from)
                        .await
                } else {
                    let axis = cast_axis(key, self.tensor.ndim())?;

                    (self.reduce)(self.tensor.clone(), axis)
                        .map(Tensor::from)
                        .map(Collection::from)
                        .map(State::from)
                }
            })
        }))
    }
}

struct TensorHandler<T> {
    tensor: T,
}

impl<'a, T: 'a> Handler<'a> for TensorHandler<T>
where
    T: TensorAccess
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorTransform
        + Clone
        + Send
        + Sync,
    <T as TensorTransform>::Slice: TensorAccess + Send,
    Tensor: From<<T as TensorTransform>::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                debug!("GET Tensor: {}", key);
                let bounds = cast_bounds(self.tensor.shape(), key)?;
                self.tensor
                    .slice(bounds)
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            debug!("PUT Tensor: {} <- {}", key, value);
            Box::pin(write(self.tensor, txn, key, value))
        }))
    }
}

impl<T> From<T> for TensorHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct UnaryHandler {
    tensor: Tensor,
    op: fn(&Tensor) -> TCResult<Tensor>,
}

impl UnaryHandler {
    fn new(tensor: Tensor, op: fn(&Tensor) -> TCResult<Tensor>) -> Self {
        Self { tensor, op }
    }
}

impl<'a> Handler<'a> for UnaryHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let tensor = if key.is_none() {
                    self.tensor
                } else {
                    let bounds = cast_bounds(self.tensor.shape(), key.into())?;
                    self.tensor.slice(bounds)?
                };

                (self.op)(&tensor).map(Collection::from).map(State::from)
            })
        }))
    }
}

struct UnaryHandlerAsync<F: Send> {
    tensor: Tensor,
    op: fn(Tensor, Txn) -> F,
}

impl<'a, F: Send> UnaryHandlerAsync<F> {
    fn new(tensor: Tensor, op: fn(Tensor, Txn) -> F) -> Self {
        Self { tensor, op }
    }
}

impl<'a, F> Handler<'a> for UnaryHandlerAsync<F>
where
    F: Future<Output = TCResult<bool>> + Send + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let txn = txn.clone();

                if key.is_none() {
                    (self.op)(self.tensor, txn).map_ok(State::from).await
                } else {
                    let bounds = cast_bounds(self.tensor.shape(), key.into())?;
                    let slice = self.tensor.slice(bounds)?;
                    (self.op)(slice, txn).map_ok(State::from).await
                }
            })
        }))
    }
}

impl<B: DenseWrite<fs::File<Array>, fs::File<Node>, fs::Dir, Txn>> Route for DenseTensor<B> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

impl<A: SparseWrite<fs::File<Array>, fs::File<Node>, fs::Dir, Txn>> Route for SparseTensor<A> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

impl Route for Tensor {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

fn route<'a, T>(tensor: &'a T, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>
where
    T: TensorAccess
        + TensorInstance
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorCompare<Tensor, Compare = Tensor, Dense = Tensor>
        + TensorBoolean<Tensor, Combine = Tensor>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorMath<fs::Dir, Tensor, Combine = Tensor>
        + TensorReduce<fs::Dir, Txn = Txn>
        + TensorTransform
        + TensorUnary<fs::Dir, Txn = Txn>
        + Clone
        + Send
        + Sync,
    Collection: From<T>,
    Tensor: From<T>,
    Tensor: From<<T as TensorInstance>::Dense> + From<<T as TensorInstance>::Sparse>,
    Tensor: From<<T as TensorReduce<fs::Dir>>::Reduce>,
    Tensor: From<<T as TensorTransform>::Cast>,
    Tensor: From<<T as TensorTransform>::Expand>,
    Tensor: From<<T as TensorTransform>::Flip>,
    Tensor: From<<T as TensorTransform>::Slice>,
    Tensor: From<<T as TensorTransform>::Transpose>,
    <T as TensorTransform>::Slice: TensorAccess + Send + 'a,
{
    if path.is_empty() {
        Some(Box::new(TensorHandler::from(tensor.clone())))
    } else if path.len() == 1 {
        match path[0].as_str() {
            // attributes
            "ndim" => {
                return Some(Box::new(AttributeHandler::from(Value::Number(
                    (tensor.ndim() as u64).into(),
                ))))
            }

            "shape" => {
                return Some(Box::new(AttributeHandler::from(
                    tensor
                        .shape()
                        .iter()
                        .map(|dim| Number::from(*dim))
                        .collect::<Tuple<Value>>(),
                )))
            }

            // reduce ops (which require borrowing)
            "product" => {
                return Some(Box::new(ReduceHandler::new(
                    tensor,
                    TensorReduce::product,
                    TensorReduce::product_all,
                )))
            }
            "sum" => {
                return Some(Box::new(ReduceHandler::new(
                    tensor,
                    TensorReduce::sum,
                    TensorReduce::sum_all,
                )))
            }
            _ => {}
        };

        let tensor = tensor.clone();

        match path[0].as_str() {
            // to stream
            "elements" => Some(Box::new(ElementsHandler::new(tensor))),

            // views
            "dense" => {
                return Some(Box::new(SelfHandlerOwned::from(Tensor::from(
                    tensor.into_dense(),
                ))));
            }

            "sparse" => {
                return Some(Box::new(SelfHandlerOwned::from(Tensor::from(
                    tensor.into_sparse(),
                ))));
            }

            // boolean ops
            "and" => Some(Box::new(DualHandler::new(
                tensor,
                TensorBoolean::and,
                TensorBooleanConst::and_const,
            ))),
            "or" => Some(Box::new(DualHandler::new(
                tensor,
                TensorBoolean::or,
                TensorBooleanConst::or_const,
            ))),
            "xor" => Some(Box::new(DualHandler::new(
                tensor,
                TensorBoolean::xor,
                TensorBooleanConst::xor_const,
            ))),

            // comparison ops
            "eq" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::eq,
                TensorCompareConst::eq_const,
            ))),
            "gt" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::gt,
                TensorCompareConst::gt_const,
            ))),
            "gte" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::gte,
                TensorCompareConst::gte_const,
            ))),
            "lt" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::lt,
                TensorCompareConst::lt_const,
            ))),
            "lte" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::lte,
                TensorCompareConst::lte_const,
            ))),
            "ne" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::ne,
                TensorCompareConst::ne_const,
            ))),

            // unary ops
            "abs" => Some(Box::new(UnaryHandler::new(tensor.into(), TensorUnary::abs))),
            "all" => Some(Box::new(UnaryHandlerAsync::new(
                tensor.into(),
                TensorUnary::all,
            ))),
            "any" => Some(Box::new(UnaryHandlerAsync::new(
                tensor.into(),
                TensorUnary::any,
            ))),
            "exp" => Some(Box::new(UnaryHandler::new(tensor.into(), TensorUnary::exp))),
            "not" => Some(Box::new(UnaryHandler::new(tensor.into(), TensorUnary::not))),

            // basic math
            "add" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::add,
                TensorMathConst::add_const,
            ))),
            "div" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::div,
                TensorMathConst::div_const,
            ))),
            "mul" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::mul,
                TensorMathConst::mul_const,
            ))),
            "pow" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::pow,
                TensorMathConst::pow_const,
            ))),
            "sub" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::sub,
                TensorMathConst::sub_const,
            ))),

            // transforms
            "cast" => Some(Box::new(CastHandler::from(tensor))),
            "flip" => Some(Box::new(FlipHandler::from(tensor))),
            "expand_dims" => Some(Box::new(ExpandHandler::from(tensor))),
            "transpose" => Some(Box::new(TransposeHandler::from(tensor))),

            _ => None,
        }
    } else {
        None
    }
}

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return None;
        }

        match path[0].as_str() {
            "dense" => TensorType::Dense.route(&path[1..]),
            "sparse" => TensorType::Sparse.route(&path[1..]),
            "copy_from" if path.len() == 1 => Some(Box::new(CopyFromHandler)),
            "einsum" if path.len() == 1 => Some(Box::new(EinsumHandler)),
            _ => None,
        }
    }
}

async fn constant(
    txn: &Txn,
    shape: Shape,
    value: Number,
) -> TCResult<DenseTensor<DenseTensorFile>> {
    let file = create_file(txn).await?;
    DenseTensor::constant(file, *txn.id(), shape, value).await
}

async fn write<T>(tensor: T, txn: &Txn, key: Value, value: State) -> TCResult<()>
where
    T: TensorAccess
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorTransform
        + Clone,
    <T as TensorTransform>::Slice: TensorAccess + Send,
{
    debug!("write {} to {}", value, key);
    let bounds = cast_bounds(tensor.shape(), key)?;

    match value {
        State::Collection(Collection::Tensor(value)) => {
            tensor.write(txn.clone(), bounds, value).await
        }
        State::Scalar(scalar) => {
            let value =
                scalar.try_cast_into(|v| TCError::bad_request("invalid tensor element", v))?;

            tensor.write_value(*txn.id(), bounds, value).await
        }
        other => Err(TCError::bad_request(
            "cannot write this value to tensor",
            other,
        )),
    }
}

async fn create_file(txn: &Txn) -> TCResult<fs::File<Array>> {
    txn.context()
        .create_file_unique(*txn.id(), TensorType::Dense)
        .await
}

fn cast_bound(dim: u64, bound: Value) -> TCResult<u64> {
    let bound = i64::try_cast_from(bound, |v| TCError::bad_request("invalid bound", v))?;
    if bound.abs() as u64 > dim {
        return Err(TCError::bad_request(
            format!("Index out of bounds for dimension {}", dim),
            bound,
        ));
    }

    if bound < 0 {
        Ok(dim - bound.abs() as u64)
    } else {
        Ok(bound as u64)
    }
}

fn cast_axis(axis: Value, ndim: usize) -> TCResult<usize> {
    debug!("cast axis {} with ndim {}", axis, ndim);

    if axis.is_none() {
        Ok(ndim)
    } else {
        let axis: Number =
            axis.try_cast_into(|v| TCError::bad_request("invalid tensor axis", v))?;

        if axis >= (ndim as u64).into() {
            Err(TCError::unsupported(format!(
                "axis {} is out of bounds for Tensor with {} dimensions",
                axis, ndim
            )))
        } else if axis >= 0.into() {
            Ok(axis.cast_into())
        } else {
            Ok(ndim - usize::cast_from(axis.abs()))
        }
    }
}

fn cast_range(dim: u64, range: Range) -> TCResult<AxisBounds> {
    debug!("cast range from {} with dimension {}", range, dim);

    let start = match range.start {
        Bound::Un => 0,
        Bound::In(start) => cast_bound(dim, start)?,
        Bound::Ex(start) => cast_bound(dim, start)? + 1,
    };

    let end = match range.end {
        Bound::Un => dim,
        Bound::In(end) => cast_bound(dim, end)? + 1,
        Bound::Ex(end) => cast_bound(dim, end)?,
    };

    if end >= start {
        Ok(AxisBounds::In(start..end))
    } else {
        Err(TCError::bad_request(
            "invalid range",
            Tuple::from(vec![start, end]),
        ))
    }
}

pub fn cast_bounds(shape: &Shape, value: Value) -> TCResult<Bounds> {
    debug!("tensor bounds from {} (shape is {})", value, shape);

    match value {
        Value::None => Ok(Bounds::all(shape)),
        Value::Number(i) => {
            let bound = cast_bound(shape[0], i.into())?;
            Ok(Bounds::from(vec![bound]))
        }
        Value::Tuple(range) if range.matches::<(Bound, Bound)>() => {
            if shape.is_empty() {
                return Err(TCError::bad_request(
                    "empty Tensor has no valid bounds, but found",
                    range,
                ));
            }

            let range = range.opt_cast_into().unwrap();
            Ok(Bounds::from(vec![cast_range(shape[0], range)?]))
        }
        Value::Tuple(bounds) => {
            if bounds.len() > shape.len() {
                return Err(TCError::unsupported(format!(
                    "tensor of shape {} does not support bounds with {} axes",
                    shape,
                    bounds.len()
                )));
            }

            let mut axes = Vec::with_capacity(shape.len());

            for (axis, bound) in bounds.into_inner().into_iter().enumerate() {
                debug!(
                    "bound for axis {} with dimension {} is {}",
                    axis, shape[axis], bound
                );

                let bound = if bound.is_none() {
                    AxisBounds::all(shape[axis])
                } else if bound.matches::<Range>() {
                    let range = Range::opt_cast_from(bound).unwrap();
                    cast_range(shape[axis], range)?
                } else if bound.matches::<Vec<u64>>() {
                    bound.opt_cast_into().map(AxisBounds::Of).unwrap()
                } else if let Value::Number(value) = bound {
                    cast_bound(shape[axis], value.into()).map(AxisBounds::At)?
                } else {
                    return Err(TCError::bad_request(
                        format!("invalid bound for axis {}", axis),
                        bound,
                    ));
                };

                axes.push(bound);
            }

            Ok(Bounds::from(axes))
        }
        other => Err(TCError::bad_request("invalid tensor bounds", other)),
    }
}
