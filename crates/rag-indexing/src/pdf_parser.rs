//! PDF文档处理
//!
//! 多模态内容提取：文本，图像，表格等

use std::collections::HashMap;
use std::path::Path;
use lopdf::{Dictionary, Document, Object, ObjectId};
use tracing::{info, warn};

use anyhow::Result;

#[derive(Debug,Clone)]
pub enum ElementType {
    Text(String),
    Image(ImageData),
    Table(TableData),
}

#[derive(Debug,Clone)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub bbox: Option<[f32; 4]>,
}

#[derive(Debug,Clone)]
pub struct TableData {
    pub rows: Vec<Vec<String>>,
    pub bbox: Option<[f32; 4]>,
}

#[derive(Debug,Clone)]
pub struct PDFElement {
    pub element_type: ElementType,
    pub page_number: usize,
    pub bbox: [f32; 4],
    pub metadata: HashMap<String, String>,
}

pub struct PDFParser;

impl PDFParser { 
    
    /// 处理PDF文档
    pub async fn parse_pdf(&self, pdf_path: &Path) -> Result<Vec<PDFElement>> {
        info!("开始处理PDF文档: {:?}", pdf_path);

        let doc = Document::load(pdf_path)
            .map_err(|e| anyhow::anyhow!("加载PDF文档失败: {:?}", e))?;

        let mut elements = Vec::new();
        
        for page_number in 0..doc.get_pages().len() {
            info!("处理第 {} 页", page_number + 1);
            let page_elements = self.parse_page(&doc, page_number as u32).await?;
            elements.extend(page_elements);
        }

        info!("完成PDF文档处理，共提取 {} 个元素", elements.len());
        Ok(elements)

    }

    async fn parse_page(&self, doc: &Document, page_number: u32) -> Result<Vec<PDFElement>> { 
        let mut elements = Vec::new();

        let pages = doc.get_pages();

        let object_id: ObjectId= *pages.get(&page_number)
            .ok_or_else(|| anyhow::anyhow!("找不到指定页"))?;

        let page_obj = doc.get_object(object_id)?;
        let page_dict = page_obj.as_dict()?;

        if let Ok(text_elements) = self.extract_text(page_dict, page_number) {
            elements.extend(text_elements);
        }

        // if let Ok(image_elements) = self.extract_image_elements(doc, *page).await {
        //     elements.extend(image_elements);
        // }

        

        Ok(elements)
    }

    fn extract_text(&self, page: &Dictionary, page_id: u32) -> Result<Vec<PDFElement>> {
        let mut elements = Vec::new();
        if let Some(content_stream) = page.get("Contents")? {
            match content_stream {
                Object::Stream(stream) => {
                    let text_content = self.parse_text_stream(stream)?;
                    let text_chunks = self.split_text_into_chunks(&text_content);
                    
                    for chunk in text_chunks {
                        elements.push(PDFElement {
                            element_type: ElementType::Text(chunk.text),
                            page_number: page_num,
                            bbox: chunk.bbox,
                            metadata: chunk.metadata,
                        });
                    }
                }
                Object::Array(arr) => {
                    // 处理多个内容流
                    for obj in arr {
                        if let Object::Stream(stream) = obj {
                            let text_content = self.parse_text_stream(stream)?;
                            let text_chunks = self.split_text_into_chunks(&text_content);
                            
                            for chunk in text_chunks {
                                elements.push(PDFElement {
                                    element_type: ElementType::Text(chunk.text),
                                    page_number: page_num,
                                    bbox: chunk.bbox,
                                    metadata: chunk.metadata,
                                });
                            }
                        }
                    }
                }
                _ => warn!("未知的文本内容类型"),
            }
        }

        Ok(elements)
        
    }

    struct TextChunk {
        text: String,
        bbox: Option<[f32; 4]>,
        metadata: HashMap<String, String>,
    }

    /// 将文本分割成合理的块
    fn split_text_into_chunks(&self, text: &str) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        
        // 按段落分割
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        
        for paragraph in paragraphs {
            let trimmed = paragraph.trim();
            if !trimmed.is_empty() {
                // 检查是否为标题（简单启发式）
                let is_header = trimmed.len() < 100 && 
                    trimmed.chars().all(|c| c.is_ascii_punctuation() || c.is_ascii_alphanumeric()) &&
                    !trimmed.ends_with('.');
                
                let metadata = if is_header {
                    HashMap::from([("type".to_string(), "header".to_string())])
                } else {
                    HashMap::from([("type".to_string(), "paragraph".to_string())])
                };
                
                chunks.push(TextChunk {
                    text: trimmed.to_string(),
                    bbox: None,
                    metadata,
                });
            }
        }

        chunks
    }

}