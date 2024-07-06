import React, { useState, useEffect } from 'react';
import { ARCanvas, ARScene } from 'react-ar';

function GalacticNexusAR() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('/api/v1/galactic_nexus')
      .then(response => response.json())
      .then(data => setData(data));
  }, []);

  return (
    <ARCanvas>
      <ARScene>
        {data.map(item => (
          <ARObject key={item.id} position={[item.x, item.y, item.z]}>
            <ARText>{item.name}</ARText>
          </ARObject>
        ))}
      </ARScene>
    </ARCanvas>
  );
}

export default GalacticNexusAR;
