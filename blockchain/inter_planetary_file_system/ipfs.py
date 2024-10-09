const ipfs = require('ipfs');

// Create a new IPFS node
const node = new ipfs();

// Function to add a file to IPFS
async function addFile(file) {
    const fileBuffer = await file.arrayBuffer();
    const fileHash = await node.add(fileBuffer);
    return fileHash;
}

// Function to get a file from IPFS
async function getFile(fileHash) {
    const fileBuffer = await node.get(fileHash);
    return fileBuffer;
}

// Add a file to IPFS
const file = new File(['Hello, world!'], 'hello.txt', { type: 'text/plain' });
const fileHash = await addFile(file);
console.log(fileHash);

// Get a file from IPFS
const fileBuffer = await getFile(fileHash);
console.log(fileBuffer);
