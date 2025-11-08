pub mod pgvector;

use sqlx::FromRow;
use anyhow::Result;
use chrono::{DateTime, Utc};
use async_trait::async_trait;
use serde_json::Value as JsonValue;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct VectorRecord {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: JsonValue,
    pub text: Option<String>,
    pub createat: Option<DateTime<Utc>>,
    pub updateat: Option<DateTime<Utc>>,
}

#[async_trait]
pub trait VectorStore {
    
    async fn add_vectors(&self, vectors: Vec<VectorRecord>) -> Result<()>;

    async fn upsert_vectors(&self, vector: Vec<VectorRecord>) -> Result<()>;

    async fn delete_vector(&self, ids: Vec<String>) -> Result<()>;

    async fn search(&self) -> Result<Vec<VectorRecord>>;

}