use std::collections::{HashMap, HashSet};

pub(crate) struct Graph {
    /// adjacency list?
    adj: HashMap<usize, HashSet<usize>>,
    nodes: HashSet<usize>,
}

impl Graph {
    pub(crate) fn new() -> Self {
        Self {
            adj: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    pub(crate) fn add_node(&mut self, node: usize) {
        self.adj.entry(node).or_insert(HashSet::new());
        self.nodes.insert(node);
    }

    pub(crate) fn add_edge(&mut self, i: usize, j: usize) {
        // for some unholy reason they don't just call add_node here
        self.add_node(i);
        self.add_node(j);
        // this is pretty strictly unnecessary, but I guess it's okay. I
        // should be able to get_mut instead since I know the or_insert
        // already happened in add_node
        self.adj
            .entry(i)
            .and_modify(|h| {
                h.insert(j);
            })
            .or_insert(HashSet::new());
        self.adj
            .entry(j)
            .and_modify(|h| {
                h.insert(i);
            })
            .or_insert(HashSet::new());
    }

    pub(crate) fn edges(&self) -> Vec<(usize, usize)> {
        let mut seen = HashSet::new();
        let mut ret = Vec::new();
        for (n, nbrs) in &self.adj {
            for nbr in nbrs {
                if !seen.contains(&nbr) {
                    ret.push((*n, *nbr));
                }
            }
            seen.insert(n);
        }
        ret
    }

    pub(crate) fn nodes(&self) -> impl Iterator<Item = &usize> {
        self.nodes.iter()
    }

    pub(crate) fn neighbors(
        &self,
        node: &usize,
    ) -> impl Iterator<Item = &usize> {
        self.adj[node].iter()
    }
}
