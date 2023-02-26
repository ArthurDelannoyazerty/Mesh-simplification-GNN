import React, { useState } from 'react';
import { Alert, Button, Card } from 'react-bootstrap';
import { Row, Col } from 'react-bootstrap';


const NotFound = () => {
    const [show, setShow] = useState(true);


  return (
    <div>
        <h1>
            not found
        </h1>
  <img src='../assets/notfound.jfif'></img>
    </div>
  );
}

export default NotFound;
