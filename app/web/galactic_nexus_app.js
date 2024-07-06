import React, { useState, useEffect } from 'react';
import axios from 'axios';

function GalacticNexusApp() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get('/api/v1/galactic_nexus')
      .then(response => {
        setData(response.data);
      })
      .catch(error => {
        console.error(error);
      });
  }, []);

  return (
    <div>
      <h1>Galactic Nexus App</h1>
      <ul>
        {data.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}

export default GalacticNexusApp;
