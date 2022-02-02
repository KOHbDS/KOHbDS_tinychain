use std::iter::FromIterator;
use std::ops;

use futures::Stream;

use tc_error::TCResult;

use crate::{AxisBounds, Coord, DenseTensorBase};

#[derive(Clone)]
pub enum IndexBounds<FD, FS, D, T> {
    Bound(AxisBounds),
    Index(DenseTensorBase<FD, FS, D, T>),
}

/// A helper struct for using one `Tensor` to index another
pub struct Index<FD, FS, D, T> {
    axes: Vec<IndexBounds<FD, FS, D, T>>,
}

impl<FD, FS, D, T> Index<FD, FS, D, T> {
    async fn coords(self) -> impl Stream<Item = TCResult<Coord>> {
        todo!("cartesian product of coordinate streams")
    }
}

impl<FD, FS, D, T> From<Vec<IndexBounds<FD, FS, D, T>>> for Index<FD, FS, D, T> {
    fn from(axes: Vec<IndexBounds<FD, FS, D, T>>) -> Self {
        Self { axes }
    }
}
