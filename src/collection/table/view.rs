use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use futures::future;
use futures::stream::{self, StreamExt, TryStreamExt};

use crate::class::{TCBoxTryFuture, TCResult, TCStream};
use crate::collection::btree::{BTreeFile, BTreeRange};
use crate::collection::schema::{Column, IndexSchema, Row};
use crate::error;
use crate::transaction::{Txn, TxnId};
use crate::value::{Value, ValueId};

use super::bounds::{self, Bounds};
use super::index::TableBase;
use super::{Selection, Table};

const ERR_AGGREGATE_SLICE: &str = "Table aggregate does not support slicing. \
Consider aggregating a slice of the source table.";
const ERR_AGGREGATE_NESTED: &str = "It doesn't make sense to aggregate an aggregate table view. \
Consider aggregating the source table directly.";
const ERR_LIMITED_ORDER: &str = "Cannot order a limited selection. \
Consider ordering the source or indexing the selection.";
const ERR_LIMITED_REVERSE: &str = "Cannot reverse a limited selection. \
Consider reversing a slice before limiting";

#[derive(Clone)]
pub struct Aggregate {
    source: Box<Table>,
    columns: Vec<ValueId>,
}

impl Aggregate {
    pub fn new(source: Table, columns: Vec<ValueId>) -> TCResult<Aggregate> {
        let source = Box::new(source.order_by(columns.to_vec(), false)?);
        Ok(Aggregate { source, columns })
    }
}

impl Selection for Aggregate {
    type Stream = TCStream<Vec<Value>>;

    fn key(&'_ self) -> &'_ [Column] {
        self.source.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.source.values()
    }

    fn group_by(&self, _columns: Vec<ValueId>) -> TCResult<Aggregate> {
        Err(error::unsupported(ERR_AGGREGATE_NESTED))
    }

    fn order_by(&self, columns: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        let source = Box::new(self.source.order_by(columns, reverse)?);
        Ok(Aggregate {
            source,
            columns: self.columns.to_vec(),
        }
        .into())
    }

    fn reversed(&self) -> TCResult<Table> {
        let columns = self.columns.to_vec();
        let reversed = self
            .source
            .reversed()
            .map(Box::new)
            .map(|source| Aggregate { source, columns })?;
        Ok(reversed.into())
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let first = self
                .source
                .clone()
                .stream(txn_id.clone())
                .await?
                .next()
                .await;
            let first = if let Some(first) = first {
                first
            } else {
                let stream: TCStream<Vec<Value>> = Box::pin(stream::empty());
                return Ok(stream);
            };

            let left = stream::once(future::ready(first))
                .chain(self.source.clone().stream(txn_id.clone()).await?);
            let right = self.source.clone().stream(txn_id).await?;
            let aggregate = left.zip(right).filter_map(|(l, r)| {
                if l == r {
                    future::ready(None)
                } else {
                    future::ready(Some(r))
                }
            });
            let aggregate: TCStream<Vec<Value>> = Box::pin(aggregate);

            Ok(aggregate)
        })
    }

    fn validate_bounds(&self, _bounds: &Bounds) -> TCResult<()> {
        Err(error::unsupported(ERR_AGGREGATE_SLICE))
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        self.source.validate_order(order)
    }
}

#[derive(Clone)]
pub struct ColumnSelection {
    source: Box<Table>,
    schema: IndexSchema,
    columns: Vec<ValueId>,
    indices: Vec<usize>,
}

impl<T: Into<Table>> TryFrom<(T, Vec<ValueId>)> for ColumnSelection {
    type Error = error::TCError;

    fn try_from(params: (T, Vec<ValueId>)) -> TCResult<ColumnSelection> {
        let (source, columns) = params;
        let source: Table = source.into();

        let column_set: HashSet<&ValueId> = columns.iter().collect();
        if column_set.len() != columns.len() {
            return Err(error::bad_request(
                "Tried to select duplicate column",
                columns
                    .iter()
                    .map(|name| name.to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        let mut indices: Vec<usize> = Vec::with_capacity(columns.len());
        let mut schema: Vec<Column> = Vec::with_capacity(columns.len());
        let source_schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();
        let mut source_columns: HashMap<ValueId, Column> = source_schema.into();

        for (i, name) in columns.iter().enumerate() {
            let column = source_columns
                .remove(name)
                .ok_or_else(|| error::not_found(name))?;
            indices.push(i);
            schema.push(column);
        }

        Ok(ColumnSelection {
            source: Box::new(source),
            schema: (vec![], schema).into(),
            columns,
            indices,
        })
    }
}

impl Selection for ColumnSelection {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        Box::pin(async move { self.source.clone().count(txn_id).await })
    }

    fn order_by(&self, order: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        self.validate_order(&order)?;

        let source = self.source.order_by(order, reverse).map(Box::new)?;

        Ok(ColumnSelection {
            source,
            schema: self.schema.clone(),
            columns: self.columns.to_vec(),
            indices: self.indices.to_vec(),
        }
        .into())
    }

    fn reversed(&self) -> TCResult<Table> {
        self.source
            .reversed()?
            .select(self.columns.to_vec())
            .map(|s| s.into())
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.schema.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.schema.values()
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let indices = self.indices.to_vec();
            let selected = self.source.clone().stream(txn_id).await?.map(move |row| {
                let selection: Vec<Value> = indices.iter().map(|i| row[*i].clone()).collect();
                selection
            });
            let selected: TCStream<Vec<Value>> = Box::pin(selected);
            Ok(selected)
        })
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let bounds_columns: HashSet<ValueId> = bounds.keys().cloned().collect();
        let selected: HashSet<ValueId> = self
            .schema
            .columns()
            .iter()
            .map(|c| c.name())
            .cloned()
            .collect();
        let mut unknown: HashSet<&ValueId> = selected.difference(&bounds_columns).collect();
        if !unknown.is_empty() {
            let unknown: Vec<String> = unknown.drain().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Tried to slice by unselected columns",
                unknown.join(", "),
            ));
        }

        self.source.validate_bounds(bounds)
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        let order_columns: HashSet<ValueId> = order.iter().cloned().collect();
        let selected: HashSet<ValueId> = self
            .schema
            .columns()
            .iter()
            .map(|c| c.name())
            .cloned()
            .collect();
        let mut unknown: HashSet<&ValueId> = selected.difference(&order_columns).collect();
        if !unknown.is_empty() {
            let unknown: Vec<String> = unknown.drain().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Tried to order by unselected columns",
                unknown.join(", "),
            ));
        }

        self.source.validate_order(order)
    }
}

#[derive(Clone)]
pub struct IndexSlice {
    source: Arc<BTreeFile>,
    schema: IndexSchema,
    bounds: Bounds,
    range: BTreeRange,
    reverse: bool,
}

impl IndexSlice {
    pub fn all(source: Arc<BTreeFile>, schema: IndexSchema, reverse: bool) -> IndexSlice {
        IndexSlice {
            source,
            schema,
            bounds: bounds::all(),
            range: BTreeRange::all(),
            reverse,
        }
    }

    pub fn new(
        source: Arc<BTreeFile>,
        schema: IndexSchema,
        bounds: Bounds,
    ) -> TCResult<IndexSlice> {
        let columns = schema.columns();

        assert!(source.schema() == &columns[..]);

        bounds::validate(&bounds, &columns)?;
        let range = bounds::btree_range(&bounds, &columns)?;

        Ok(IndexSlice {
            source,
            schema,
            bounds,
            range,
            reverse: false,
        })
    }

    pub fn schema(&'_ self) -> &'_ IndexSchema {
        &self.schema
    }

    pub fn into_reversed(mut self) -> IndexSlice {
        self.reverse = !self.reverse;
        self
    }

    pub fn slice_index(&self, bounds: Bounds) -> TCResult<IndexSlice> {
        let columns = self.schema().columns();
        let outer = bounds::btree_range(&self.bounds, &columns)?;
        let inner = bounds::btree_range(&bounds, &columns)?;

        if outer.contains(&inner, self.schema.data_types())? {
            let mut slice = self.clone();
            slice.bounds = bounds;
            Ok(slice)
        } else {
            Err(error::bad_request(
                &format!(
                    "IndexSlice with bounds {} does not contain",
                    bounds::format(&self.bounds)
                ),
                bounds::format(&bounds),
            ))
        }
    }
}

impl Selection for IndexSlice {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        self.source.clone().len(txn_id, self.range.clone().into())
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { self.source.delete(&txn_id, self.range.into()).await })
    }

    fn order_by(&self, order: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        if self.schema.starts_with(&order) {
            if reverse {
                self.reversed()
            } else {
                Ok(self.clone().into())
            }
        } else {
            let order: Vec<String> = order.iter().map(String::from).collect();
            Err(error::bad_request(
                &format!("Index with schema {} does not support order", &self.schema),
                order.join(", "),
            ))
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(self.clone().into_reversed().into())
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.schema.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.schema.values()
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            self.source
                .clone()
                .slice(txn_id.clone(), self.range.clone().into())
                .await
        })
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            self.source
                .update(
                    txn.id(),
                    &self.range.into(),
                    &self.schema.row_into_values(value, true)?,
                )
                .await
        })
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let schema = self.schema();
        let outer = bounds::btree_range(&self.bounds, &schema.columns())?;
        let inner = bounds::btree_range(&bounds, &schema.columns())?;
        outer.contains(&inner, schema.data_types()).map(|_| ())
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        if self.schema.starts_with(order) {
            Ok(())
        } else {
            let order: Vec<String> = order.iter().map(String::from).collect();
            Err(error::bad_request(
                &format!("Index with schema {} does not support order", &self.schema),
                order.join(", "),
            ))
        }
    }
}

#[derive(Clone)]
pub struct Limited {
    source: Box<Table>,
    limit: usize,
}

impl TryFrom<(Table, u64)> for Limited {
    type Error = error::TCError;

    fn try_from(params: (Table, u64)) -> TCResult<Limited> {
        let (source, limit) = params;
        let limit: usize = limit.try_into().map_err(|_| {
            error::internal("This host architecture does not support a 64-bit stream limit")
        })?;

        Ok(Limited {
            source: Box::new(source),
            limit,
        })
    }
}

impl Selection for Limited {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        Box::pin(async move {
            let source_count = self.source.count(txn_id).await?;
            Ok(u64::min(source_count, self.limit as u64))
        })
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let source = self.source.clone();
            let schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();
            self.stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| source.delete_row(&txn_id, row))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn order_by(&self, _order: Vec<ValueId>, _reverse: bool) -> TCResult<Table> {
        Err(error::unsupported(ERR_LIMITED_ORDER))
    }

    fn reversed(&self) -> TCResult<Table> {
        Err(error::unsupported(ERR_LIMITED_REVERSE))
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.source.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.source.values()
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let rows = self.source.clone().stream(txn_id).await?;
            let rows: TCStream<Vec<Value>> = Box::pin(rows.take(self.limit));
            Ok(rows)
        })
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        self.source.validate_bounds(bounds)
    }

    fn validate_order(&self, _order: &[ValueId]) -> TCResult<()> {
        Err(error::unsupported(ERR_LIMITED_ORDER))
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let source = self.source.clone();
            let schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();
            let txn_id = txn.id().clone();
            self.stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| source.update_row(txn_id.clone(), row, value.clone()))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }
}

#[derive(Clone)]
pub enum MergeSource {
    Table(TableSlice),
    Merge(Arc<Merged>),
}

impl MergeSource {
    fn into_reversed(self) -> MergeSource {
        match self {
            Self::Table(table_slice) => Self::Table(table_slice.into_reversed()),
            Self::Merge(merged) => Self::Merge(merged.as_reversed()),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Table> {
        match self {
            Self::Table(table) => table.slice(bounds),
            Self::Merge(merged) => merged.slice(bounds),
        }
    }
}

#[derive(Clone)]
pub struct Merged {
    left: MergeSource,
    right: IndexSlice,
}

impl Merged {
    pub fn new(left: MergeSource, right: IndexSlice) -> Merged {
        Merged { left, right }
    }

    fn as_reversed(self: Arc<Self>) -> Arc<Self> {
        Arc::new(Merged {
            left: self.left.clone().into_reversed(),
            right: self.right.clone().into_reversed(),
        })
    }
}

impl Selection for Merged {
    type Stream = TCStream<Vec<Value>>;

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();
            self.clone()
                .stream(txn_id.clone())
                .await?
                .map(|values| schema.values_into_row(values))
                .map_ok(|row| self.delete_row(&txn_id, row))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        match &self.left {
            MergeSource::Table(table) => table.delete_row(txn_id, row),
            MergeSource::Merge(merged) => merged.delete_row(txn_id, row),
        }
    }

    fn order_by(&self, columns: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        match &self.left {
            MergeSource::Merge(merged) => merged.order_by(columns, reverse),
            MergeSource::Table(table_slice) => table_slice.order_by(columns, reverse),
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(Merged {
            left: self.left.clone().into_reversed(),
            right: self.right.clone().into_reversed(),
        }
        .into())
    }

    fn key(&'_ self) -> &'_ [Column] {
        match &self.left {
            MergeSource::Table(table) => table.key(),
            MergeSource::Merge(merged) => merged.key(),
        }
    }

    fn values(&'_ self) -> &'_ [Column] {
        match &self.left {
            MergeSource::Table(table) => table.values(),
            MergeSource::Merge(merged) => merged.values(),
        }
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        // TODO: reject bounds which lie outside the bounds of the table slice

        match &self.left {
            MergeSource::Merge(merged) => merged.slice(bounds),
            MergeSource::Table(table) => table.slice(bounds),
        }
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let key_columns = self.key().to_vec();
            let key_names: Vec<ValueId> = key_columns.iter().map(|c| c.name()).cloned().collect();
            let left = self.left.clone();
            let txn_id_clone = txn_id.clone();
            let rows = self
                .right
                .select(key_names)?
                .stream(txn_id.clone())
                .await?
                .map(move |key| bounds::from_key(key, &key_columns))
                .map(move |bounds| left.clone().slice(bounds))
                .map(|slice| slice.unwrap())
                .then(move |slice| slice.stream(txn_id_clone.clone()))
                .map(|stream| stream.unwrap())
                .flatten();

            let rows: TCStream<Vec<Value>> = Box::pin(rows);
            Ok(rows)
        })
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        match &self.left {
            MergeSource::Merge(merge) => merge.validate_bounds(bounds),
            MergeSource::Table(table) => table.validate_bounds(bounds),
        }
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        match &self.left {
            MergeSource::Merge(merge) => merge.validate_order(order),
            MergeSource::Table(table) => table.validate_order(order),
        }
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();
            self.clone()
                .stream(txn.id().clone())
                .await?
                .map(|values| schema.values_into_row(values))
                .map_ok(|row| self.update_row(txn.id().clone(), row, value.clone()))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCBoxTryFuture<()> {
        match &self.left {
            MergeSource::Table(table) => table.update_row(txn_id, row, value),
            MergeSource::Merge(merged) => merged.update_row(txn_id, row, value),
        }
    }
}

#[derive(Clone)]
pub struct TableSlice {
    table: TableBase,
    bounds: Bounds,
    reversed: bool,
}

impl TableSlice {
    pub fn new(table: TableBase, bounds: Bounds) -> TCResult<TableSlice> {
        table.validate_bounds(&bounds)?;

        Ok(TableSlice {
            table,
            bounds,
            reversed: false,
        })
    }

    fn into_reversed(self) -> TableSlice {
        TableSlice {
            table: self.table,
            bounds: self.bounds,
            reversed: !self.reversed,
        }
    }
}

impl Selection for TableSlice {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        Box::pin(async move {
            let index = self.table.supporting_index(&self.bounds)?;
            index.slice(self.bounds.clone())?.count(txn_id).await
        })
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();
            self.clone()
                .stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| self.delete_row(&txn_id, row))
                .try_buffer_unordered(2)
                .fold(Ok(()), |_, r| future::ready(r))
                .await
        })
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        self.table.delete_row(txn_id, row)
    }

    fn order_by(&self, order: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        self.table.order_by(order, reverse)
    }

    fn reversed(&self) -> TCResult<Table> {
        let mut selection = self.clone();
        selection.reversed = true;
        Ok(selection.into())
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.table.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.table.values()
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        self.validate_bounds(&bounds)?;
        self.table.slice(bounds)
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            let slice = self.table.primary().slice(self.bounds.clone())?;

            slice.stream(txn_id).await
        })
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let index = self.table.supporting_index(&self.bounds)?;
        index
            .validate_slice_bounds(self.bounds.clone(), bounds.clone())
            .map(|_| ())
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        self.table.validate_order(order)
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let txn_id = txn.id().clone();
            let schema: IndexSchema = (self.key().to_vec(), self.values().to_vec()).into();
            self.clone()
                .stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| self.update_row(txn_id.clone(), row, value.clone()))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCBoxTryFuture<()> {
        self.table.update_row(txn_id, row, value)
    }
}