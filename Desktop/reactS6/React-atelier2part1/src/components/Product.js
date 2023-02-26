import React, { useState } from 'react';
import { Alert, Button, Card } from 'react-bootstrap';
import { Row, Col } from 'react-bootstrap';

export default function Product(props) {
  const [produit, setProduit] = useState(props.produit);
  const [show, setShow] = useState(false);
  const [isBestProduct, setIsBestProduct] = useState(false);

  const likeProduct = (e) => {
    e.preventDefault();
    const updatedProduit = { ...produit, like: produit.like + 1 };
    setProduit(updatedProduit);

    if (updatedProduit.like > 5) {
      setIsBestProduct(true);
    }
  };

  const buyProduct = (e) => {
    e.preventDefault();
    const updatedProduit = { ...produit, quantity: produit.quantity - 1 };
    setProduit(updatedProduit);

    setShow(true);
    setTimeout(() => {
      setShow(false);
    }, 2000);
  };

  return (
    <Card  className={isBestProduct ? 'best-product ':'' } style={{ width: '18rem', margin: '10%' }}   border="secondary">
      <Card.Img src={require('../assets/images/' + props.produit.img)} />
      <Card.Body>
        <Card.Title >{props.produit.name}</Card.Title>
        <Card.Text>
          {props.produit.description}
          <br />
          <br />
          <Row>
            <Col md={6}>{props.produit.price + 'dt'}</Col>
            <Col md={6}>{props.produit.quantity}</Col>
          </Row>
        </Card.Text>
        <Row>
          <Col md={6}>
            <Button onClick={likeProduct} >
              Like:{produit.like}
            </Button>
          </Col>
          <Col md={6}>
            <Button variant="success" onClick={buyProduct} disabled={produit.quantity === 0}>
              Buy
            </Button>
            quantity:{produit.quantity}
          </Col>
        </Row>
      </Card.Body>
      <Alert show={show} variant="primary">
        You bought the Item named {produit.name}
      </Alert>
    </Card>
  );
}
