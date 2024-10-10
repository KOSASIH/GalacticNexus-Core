// File: blockchain.rs
use {
    std::collections::{HashMap, VecDeque},
    std::sync::{Arc, Mutex},
    crypto::{Hash, Hmac},
    network::{Network, Node},
};

struct Blockchain {
    chain: Vec<Block>,
    pending_transactions: VecDeque<Transaction>,
    transaction_pool: HashMap<Hash, Transaction>,
    node: Arc<Node>,
}

impl Blockchain {
    fn new(node: Arc<Node>) -> Self {
        Self {
            chain: vec![genesis_block()],
            pending_transactions: VecDeque::new(),
            transaction_pool: HashMap::new(),
            node,
        }
    }

    fn add_block(&mut self, block: Block) {
        self.chain.push(block);
        self.pending_transactions.clear();
    }

    fn add_transaction(&mut self, transaction: Transaction) {
        self.transaction_pool.insert(transaction.hash(), transaction);
        self.pending_transactions.push_back(transaction);
    }

    fn mine_block(&mut self) -> Block {
        let transactions: Vec<_> = self.pending_transactions.drain(..).collect();
        let block = Block::new(transactions, self.chain.last().unwrap().hash());
        self.add_block(block.clone());
        block
    }
}

fn genesis_block() -> Block {
    Block {
        transactions: vec![],
        previous_hash: Hash::zero(),
        nonce: 0,
    }
}

struct Block {
    transactions: Vec<Transaction>,
    previous_hash: Hash,
    nonce: u64,
}

struct Transaction {
    from: Hash,
    to: Hash,
    amount: u64,
    hash: Hash,
}

impl Transaction {
    fn new(from: Hash, to: Hash, amount: u64) -> Self {
        let hash = Hmac::new(from, to, amount).digest();
        Self { from, to, amount, hash }
    }
}
