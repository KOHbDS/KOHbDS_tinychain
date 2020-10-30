use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use futures::TryFutureExt;

use crate::auth::Auth;
use crate::class::{Instance, State, TCBoxTryFuture};
use crate::error::{self, TCResult};
use crate::scalar::{self, Op, OpRef, Scalar, TCPath, Value, ValueInstance};
use crate::transaction::Txn;

use super::InstanceClass;

#[derive(Clone)]
pub struct ObjectInstance {
    parent: Box<State>,
    class: InstanceClass,
}

impl ObjectInstance {
    pub async fn new(
        class: InstanceClass,
        txn: Arc<Txn>,
        schema: Value,
        auth: Auth,
    ) -> TCResult<ObjectInstance> {
        let ctr = OpRef::Get((class.extends(), schema));
        let parent = txn
            .resolve(HashMap::new(), ctr.into(), auth)
            .map_ok(Box::new)
            .await?;

        Ok(ObjectInstance { parent, class })
    }

    pub fn get<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        key: Value,
        auth: Auth,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            println!("ObjectInstance::get {}: {}", path, key);

            let proto = self.class.proto().data();
            match proto.get(&path[0]) {
                Some(scalar) => match scalar {
                    Scalar::Op(op) if path.len() == 1 => match &**op {
                        Op::Def(op_def) => op_def.get(txn, key, auth, Some(self)).await,
                        other => Err(error::not_implemented(format!(
                            "ObjectInstance::get {}",
                            other
                        ))),
                    },
                    Scalar::Op(_) => Err(error::not_found(path.slice_from(1))),
                    Scalar::Value(value) => value
                        .get(path.slice_from(1), key)
                        .map(Scalar::Value)
                        .map(State::Scalar),
                    other => Err(error::not_implemented(format!(
                        "ObjectInstance::get {}",
                        other
                    ))),
                },
                None => match &*self.parent {
                    State::Object(parent) => parent.get(txn, path, key, auth).await,
                    State::Scalar(scalar) => match scalar {
                        Scalar::Value(value) => {
                            value.get(path, key).map(Scalar::Value).map(State::Scalar)
                        }
                        _ => Err(error::not_implemented("Class inheritance for Scalar")),
                    },
                    _ => Err(error::not_implemented("Class inheritance for State")),
                },
            }
        })
    }

    pub async fn post(
        &self,
        _txn: Arc<Txn>,
        path: TCPath,
        _data: scalar::Object,
        _auth: Auth,
    ) -> TCResult<State> {
        if path.is_empty() {
            Err(error::not_implemented("ObjectInstance::post"))
        } else {
            Err(error::not_found(path))
        }
    }
}

impl Instance for ObjectInstance {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl From<scalar::object::Object> for ObjectInstance {
    fn from(generic: scalar::object::Object) -> ObjectInstance {
        ObjectInstance {
            parent: Box::new(State::Scalar(Scalar::Object(generic))),
            class: InstanceClass::default(),
        }
    }
}

impl fmt::Display for ObjectInstance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}