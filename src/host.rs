use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::drive::Drive;
use crate::error;
use crate::state::TCState;
use crate::state::block::BlockContext;
use crate::state::chain::ChainContext;
use crate::state::table::TableContext;
use crate::transaction::Transaction;
use crate::value::{Link, Op};

pub struct Host {
    block_context: Arc<BlockContext>,
    chain_context: Arc<ChainContext>,
    table_context: Arc<TableContext>,
}

impl Host {
    pub fn new(workspace: Arc<Drive>) -> TCResult<Arc<Host>> {
        let block_context = BlockContext::new(workspace);
        let chain_context = ChainContext::new();
        let table_context = TableContext::new();
        Ok(Arc::new(Host {
            block_context,
            chain_context,
            table_context,
        }))
    }

    pub fn time(&self) -> u128 {
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    pub fn new_transaction(self: Arc<Self>, op: Op) -> TCResult<Arc<Transaction>> {
        Transaction::of(self, op)
    }

    pub async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<TCState> {
        match path[0].as_str() {
            "sbin" => match path[1].as_str() {
                "block" => {
                    self.block_context
                        .clone()
                        .get(txn, path.from("/sbin/block")?)
                        .await
                }
                "chain" => {
                    self.block_context
                        .clone()
                        .get(txn, path.from("/sbin/chain")?)
                        .await
                }
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }

    pub async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Link,
        state: TCState,
    ) -> TCResult<()> {
        if path.len() != 2 {
            return Err(error::not_found(path));
        }

        match path[0].as_str() {
            "sbin" => match path[1].as_str() {
                "block" => self.block_context.clone().put(txn, state).await,
                "chain" => self.block_context.clone().put(txn, state).await,
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }

    pub async fn post(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<TCState> {
        match path[0].as_str() {
            "sbin" => match path[1].as_str() {
                "table" => {
                    self.table_context
                        .clone()
                        .post(txn, path.from("/sbin/table")?)
                        .await
                }
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }
}
