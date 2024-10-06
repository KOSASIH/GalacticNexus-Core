// Import the necessary libraries
const axios = require('axios');

// Define the gnx_network functions
async function getData(url) {
  const response = await axios.get(url);
  return response.data;
}

async function postData(url, data) {
  const response = await axios.post(url, data);
  return response.data;
}

// Export the gnx_network functions
module.exports = { getData, postData };
