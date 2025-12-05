import React from 'react';
import styles from '../styles/HomePage.module.css';

function CardContainer() {
  return (
    <div className={styles.cardContainer}>
      <div className={styles.card}></div>
      <div className={styles.card}></div>
      <div className={styles.card}></div>
    </div>
  );
}

function HomePage() {
  return (
    <main className={styles.mainContent}>
      <CardContainer />
    </main>
  );
}

export default HomePage;
