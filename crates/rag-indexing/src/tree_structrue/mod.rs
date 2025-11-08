pub mod markdown_bulid;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub type NodeId = Uuid;
pub type ParentId = Option<NodeId>;
pub type ChildrenIds = Vec<NodeId>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeRelationship {
    Parent,
    Child,
    Previous,
    Next,
    Root,
    Source,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Root,
    Intermediate,
    Leaf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub document_id: String,
    pub hierarchy: Vec<String>,
    pub node_type: NodeType,
    pub chunk_size: Option<usize>,
    pub file_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    Root(RootNode),
    Intermediate(IntermediateNode),
    Leaf(LeafNode),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootNode {
    pub id: NodeId,
    pub document_id: String,
    pub relationships: HashMap<NodeRelationship, Vec<NodeId>>,
    pub metadata: NodeMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateNode {
    pub id: NodeId,
    pub title: Option<String>,
    pub relationships: HashMap<NodeRelationship, Vec<NodeId>>,
    pub metadata: NodeMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafNode {
    pub id: NodeId,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    pub relationships: HashMap<NodeRelationship, Vec<NodeId>>,
    pub metadata: NodeMetadata,
}

impl Node {
    pub fn new_root(document_id: String, file_name: Option<String>) -> Self {
        let id = Uuid::new_v4();
        let mut relationships = HashMap::new();
        relationships.insert(NodeRelationship::Root, vec![id]);

        Node::Root(RootNode {
            id,
            document_id: document_id.clone(),
            relationships,
            metadata: NodeMetadata {
                document_id,
                hierarchy: vec!["Root".to_string()],
                node_type: NodeType::Root,
                chunk_size: None,
                file_name,
            },
        })
    }

    pub fn new_intermediate(
        parent_id: NodeId,
        title: Option<String>,
        hierarchy: Vec<String>,
        document_id: String,
    ) -> Self {
        let id = Uuid::new_v4();
        let mut relationships = HashMap::new();
        relationships.insert(NodeRelationship::Parent, vec![parent_id]);

        // let mut hier = hierarchy.clone();
        // if let Some(t) = &title {
        //     hier.push(t.clone());
        // }

        Node::Intermediate(IntermediateNode {
            id,
            title,
            relationships,
            metadata: NodeMetadata {
                document_id,
                hierarchy,
                node_type: NodeType::Intermediate,
                chunk_size: None,
                file_name: None,
            },
        })
    }

    pub fn new_leaf(
        parent_id: NodeId,
        text: String,
        chunk_size: usize,
        chunk_index: usize,
        hierarchy: Vec<String>,
        document_id: String,
    ) -> Self {
        let id = Uuid::new_v4();
        let mut relationships = HashMap::new();
        relationships.insert(NodeRelationship::Parent, vec![parent_id]);

        let mut hier = hierarchy;
        hier.push(format!("chunk_{}_{}", chunk_index, chunk_size));

        Node::Leaf(LeafNode {
            id,
            text,
            embedding: None,
            relationships,
            metadata: NodeMetadata {
                document_id,
                hierarchy: hier,
                node_type: NodeType::Leaf,
                chunk_size: Some(chunk_size),
                file_name: None,
            },
        })
    }

    pub fn id(&self) -> NodeId {
        match self {
            Node::Root(n) => n.id,
            Node::Intermediate(n) => n.id,
            Node::Leaf(n) => n.id,
        }
    }

    pub fn parent_id(&self) -> Option<NodeId> {
        self.relationships()
            .get(&NodeRelationship::Parent)
            .and_then(|v| v.first().copied())
    }

    pub fn children(&self) -> &[NodeId] {
        self.relationships()
            .get(&NodeRelationship::Child)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn children_mut(&mut self) -> &mut Vec<NodeId> {
        self.relationships_mut()
            .entry(NodeRelationship::Child)
            .or_insert(vec![])
    }

    pub fn prev_id(&self) -> Option<NodeId> {
        self.relationships()
            .get(&NodeRelationship::Previous)
            .and_then(|v| v.first().copied())
    }

    pub fn next_id(&self) -> Option<NodeId> {
        self.relationships()
            .get(&NodeRelationship::Next)
            .and_then(|v| v.first().copied())
    }

    pub fn set_previous(&mut self, prev_id: Option<NodeId>) {
        let rel = self.relationships_mut();
        if let Some(id) = prev_id {
            rel.insert(NodeRelationship::Previous, vec![id]);
        } else {
            rel.remove(&NodeRelationship::Previous);
        }
    }

    pub fn set_next(&mut self, next_id: Option<NodeId>) {
        let rel = self.relationships_mut();
        if let Some(id) = next_id {
            rel.insert(NodeRelationship::Next, vec![id]);
        } else {
            rel.remove(&NodeRelationship::Next);
        }
    }

    pub fn metadata(&self) -> &NodeMetadata {
        match self {
            Node::Root(n) => &n.metadata,
            Node::Intermediate(n) => &n.metadata,
            Node::Leaf(n) => &n.metadata,
        }
    }

    pub fn metadata_mut(&mut self) -> &mut NodeMetadata {
        match self {
            Node::Root(n) => &mut n.metadata,
            Node::Intermediate(n) => &mut n.metadata,
            Node::Leaf(n) => &mut n.metadata,
        }
    }

    pub fn as_leaf(&self) -> Option<&LeafNode> {
        match self {
            Node::Leaf(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_leaf_mut(&mut self) -> Option<&mut LeafNode> {
        match self {
            Node::Leaf(n) => Some(n),
            _ => None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf(_))
    }

    fn relationships(&self) -> &HashMap<NodeRelationship, Vec<NodeId>> {
        match self {
            Node::Root(n) => &n.relationships,
            Node::Intermediate(n) => &n.relationships,
            Node::Leaf(n) => &n.relationships,
        }
    }

    fn relationships_mut(&mut self) -> &mut HashMap<NodeRelationship, Vec<NodeId>> {
        match self {
            Node::Root(n) => &mut n.relationships,
            Node::Intermediate(n) => &mut n.relationships,
            Node::Leaf(n) => &mut n.relationships,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTree {
    pub nodes: HashMap<NodeId, Node>,
    pub root: NodeId,
}

impl NodeTree {
    pub fn new(root: Node) -> Self {
        let root_id = root.id();
        let mut nodes = HashMap::new();
        nodes.insert(root_id, root);
        Self { nodes, root: root_id }
    }

    /// 添加节点，自动维护双向关系 + prev/next
    pub fn add_node(&mut self, mut child_node: Node) -> Result<()> {
        let parent_id = child_node.parent_id()
            .ok_or_else(|| anyhow!("Node must have a parent"))?;

        let parent = self.nodes.get_mut(&parent_id)
            .ok_or_else(|| anyhow!("Parent node {} not found", parent_id))?;

        // 1. 父节点添加子节点
        parent.children_mut().push(child_node.id());

        // 2. 维护 prev/next
        if let Some(last_child_id) = parent.children().iter().rev().nth(1).copied() {
            // 上一个兄弟
            if let Some(prev_node) = self.nodes.get_mut(&last_child_id) {
                prev_node.set_next(Some(child_node.id()));
            }
            child_node.set_previous(Some(last_child_id));
        }

        // 3. 插入
        self.nodes.insert(child_node.id(), child_node);
        Ok(())
    }

    /// 获取所有叶子节点
    pub fn leaf_nodes(&self) -> impl Iterator<Item = &LeafNode> {
        self.nodes.values()
            .filter_map(|n| n.as_leaf())
    }

}