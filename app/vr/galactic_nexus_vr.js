import React, { useState, useEffect } from 'react';
import { VRCanvas, VRScene } from 'react-vr';

function GalacticNexusVR() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('/api/v1/galactic_nexus')
      .then(response => response.json())
      .then(data => setData(data));
  }, []);

  return (
    <VRCanvas>
      <VRScene>
        {data.map(item => (
          <VRObject key={item.id} position={[item.x, item.y, item.z]}>
            <VRText>{item.name}</VRText>
          </VRObject>
        ))}
      </VRScene>
    </VRCanvas>
  );
}

export default GalacticNexusVR;
