use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;
use futures::future::join_all;
use futures::TryFutureExt;

use error::*;
use generic::*;
use safecast::TryCastInto;
use transact::fs::{Dir, Persist};
use transact::{Transact, TxnId};
use value::Value;

use crate::chain::{Chain, ChainType, SyncChain};
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::scalar::{OpRef, Scalar, TCRef};

pub const PATH: Label = label("cluster");

pub struct ClusterType;

impl Class for ClusterType {
    type Instance = Cluster;
}

impl fmt::Display for ClusterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Cluster")
    }
}

#[derive(Clone)]
pub struct Cluster {
    path: TCPathBuf,
    chains: Map<Chain>,
}

impl Eq for Cluster {}

impl PartialEq for Cluster {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl Hash for Cluster {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state)
    }
}

impl Cluster {
    pub async fn instantiate(
        class: InstanceClass,
        data_dir: fs::Dir,
        txn_id: TxnId,
    ) -> TCResult<InstanceExt<Cluster>> {
        let (path, proto) = class.into_inner();
        let path = path.ok_or_else(|| {
            TCError::unsupported("cluster config must specify the path of the cluster to host")
        })?;
        let path = path.into_path();

        let mut chain_schema = HashMap::new();
        let mut cluster_proto = HashMap::new();
        for (id, scalar) in proto.into_iter() {
            match scalar {
                Scalar::Ref(tc_ref) => {
                    let (ct, schema) = if let TCRef::Op(OpRef::Get((path, schema))) = *tc_ref {
                        let path: TCPathBuf = path.try_into()?;
                        let schema: Value = schema.try_into()?;

                        if let Some(ct) = ChainType::from_path(&path) {
                            (ct, schema)
                        } else {
                            return Err(TCError::bad_request(
                                "Cluster requires its mutable data to be wrapped in a chain, not",
                                path,
                            ));
                        }
                    } else {
                        return Err(TCError::bad_request("expected a Chain but found", tc_ref));
                    };

                    chain_schema.insert(id, (ct, schema));
                }
                Scalar::Op(op_def) => {
                    cluster_proto.insert(id, Scalar::Op(op_def));
                }
                other => return Err(TCError::bad_request(
                    "Cluster member must be a Chain (for mutable data), or an immutable OpDef, not",
                    other,
                )),
            }
        }

        let dir = if let Some(dir) = data_dir.find(&txn_id, &path).await? {
            match dir {
                fs::DirEntry::Dir(dir) => dir,
                _ => {
                    return Err(TCError::bad_request("there is already a file at", &path));
                }
            }
        } else {
            create_dir(data_dir, txn_id, &path).await?
        };

        let mut chains = HashMap::<Id, Chain>::new();
        for (id, (class, schema)) in chain_schema.into_iter() {
            let dir = dir.create_dir(txn_id, id.clone()).await?;
            let chain = match class {
                ChainType::Sync => {
                    let schema = schema
                        .try_cast_into(|v| TCError::bad_request("invalid Chain schema", v))?;

                    SyncChain::load(schema, dir, txn_id)
                        .map_ok(Chain::Sync)
                        .await?
                }
            };

            chains.insert(id, chain);
        }

        let cluster = Cluster {
            path: path.clone(),
            chains: chains.into(),
        };

        let class = InstanceClass::new(Some(path.into()), cluster_proto.into());

        Ok(InstanceExt::new(cluster, class))
    }

    pub fn path(&'_ self) -> &'_ [PathSegment] {
        &self.path
    }
}

impl Instance for Cluster {
    type Class = ClusterType;

    fn class(&self) -> Self::Class {
        ClusterType
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.commit(txn_id))).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join_all(self.chains.values().map(|chain| chain.finalize(txn_id))).await;
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", self.path)
    }
}

async fn create_dir(data_dir: fs::Dir, txn_id: TxnId, path: &[PathSegment]) -> TCResult<fs::Dir> {
    let mut dir = data_dir;
    for name in path {
        dir = dir.create_dir(txn_id, name.clone()).await?;
    }

    Ok(dir)
}
