pub mod qwen;
use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Network error: {0}")]
    Network(String),
    #[error("API error: {0}")]
    Api(String),
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    #[error("Invalid vector: {0}")]
    InvalidVector(String),
}

pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

/// 统一向量嵌入接口
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// 批量嵌入文本
    async fn embed(&self, texts: Vec<String>) -> EmbeddingResult<Vec<Vec<f32>>>;

    /// 获取向量维度
    fn dimension(&self) -> usize;
}