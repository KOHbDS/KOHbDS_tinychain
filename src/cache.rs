use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

pub struct Map<K: Eq + Hash, V: Hash> {
    cache: RwLock<HashMap<K, Arc<V>>>,
}

impl<K: Eq + Hash, V: Hash> Map<K, V> {
    pub fn new() -> Map<K, V> {
        Map {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.cache.read().unwrap().contains_key(key)
    }

    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.cache.read().unwrap().get(key) {
            Some(state) => Some(state.clone()),
            None => None,
        }
    }

    pub fn insert(&self, key: K, value: Arc<V>) -> Option<Arc<V>> {
        self.cache.write().unwrap().insert(key, value)
    }
}

pub struct Value<T: Copy> {
    val: RwLock<T>,
}

impl<T: Copy> Value<T> {
    pub fn of(value: T) -> Value<T> {
        Value {
            val: RwLock::new(value),
        }
    }

    pub fn get(self: Arc<Self>) -> T {
        self.val.read().unwrap().clone()
    }

    pub fn set(self: Arc<Self>, value: T) {
        *self.val.write().unwrap() = value;
    }
}
