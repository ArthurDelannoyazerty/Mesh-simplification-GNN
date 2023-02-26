import React, { useState, useEffect } from 'react';
import data from '../products.json';
import Product from './Product';
import { Container, Row , Col ,Navbar , Nav, NavLink } from 'react-bootstrap';

const  Products = () => {
  const [color, setColor] = useState("green");

  useEffect(() => {
  }, []);

  const buyProd = (produit) => {
    setColor("blue");
  }

  return (
  <>

<Navbar bg="light" variant="light">

<Container>

<Navbar.Brand to="/products">MyStore</Navbar.Brand>

<Nav className="me-auto">

<Nav.Link as={NavLink} to="/products" >Products</Nav.Link>

</Nav>

</Container>
</Navbar>


    <Container>
      <Row>
        {data.map((prod, index) => (
          <Col key={index} md={4}>
            <Product produit={prod} buyProduit={buyProd} />
          </Col>
        ))}
      </Row>
    </Container>
    </>
  );
  
}

export default Products;
