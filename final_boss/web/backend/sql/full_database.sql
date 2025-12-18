-- MySQL dump 10.13  Distrib 8.0.44, for Linux (x86_64)
--
-- Host: localhost    Database: KiThuatDuLieu
-- ------------------------------------------------------
-- Server version	8.0.44-0ubuntu0.24.04.2

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Dim_HealthCareAccess`
--
CREATE DATABASE IF NOT EXISTS KiThuatDuLieu
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE KiThuatDuLieu;


DROP TABLE IF EXISTS `Dim_HealthCareAccess`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Dim_HealthCareAccess` (
  `AccessID` int NOT NULL AUTO_INCREMENT,
  `AnyHealthcare` int DEFAULT NULL,
  `NoDocbcost` int DEFAULT NULL,
  `CholCheck` int DEFAULT NULL,
  PRIMARY KEY (`AccessID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Dim_HealthCareAccess`
--

LOCK TABLES `Dim_HealthCareAccess` WRITE;
/*!40000 ALTER TABLE `Dim_HealthCareAccess` DISABLE KEYS */;
/*!40000 ALTER TABLE `Dim_HealthCareAccess` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Dim_HealthStatus`
--

DROP TABLE IF EXISTS `Dim_HealthStatus`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Dim_HealthStatus` (
  `HealthID` int NOT NULL AUTO_INCREMENT,
  `HighBP` int DEFAULT NULL,
  `HighChol` int DEFAULT NULL,
  `BMI` float DEFAULT NULL,
  `GenHlth` int DEFAULT NULL,
  `MentHlth` int DEFAULT NULL,
  `PhysHlth` int DEFAULT NULL,
  `Stroke` int DEFAULT NULL,
  `HeartDiseaseorAttack` int DEFAULT NULL,
  PRIMARY KEY (`HealthID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Dim_HealthStatus`
--

LOCK TABLES `Dim_HealthStatus` WRITE;
/*!40000 ALTER TABLE `Dim_HealthStatus` DISABLE KEYS */;
/*!40000 ALTER TABLE `Dim_HealthStatus` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Dim_LifeStyle`
--

DROP TABLE IF EXISTS `Dim_LifeStyle`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Dim_LifeStyle` (
  `LifeStyleID` int NOT NULL AUTO_INCREMENT,
  `Smoker` int DEFAULT NULL,
  `PhysActivity` int DEFAULT NULL,
  `Fruits` int DEFAULT NULL,
  `Veggies` int DEFAULT NULL,
  `HvyAlcoholConsump` int DEFAULT NULL,
  `DiffWalk` int DEFAULT NULL,
  PRIMARY KEY (`LifeStyleID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Dim_LifeStyle`
--

LOCK TABLES `Dim_LifeStyle` WRITE;
/*!40000 ALTER TABLE `Dim_LifeStyle` DISABLE KEYS */;
/*!40000 ALTER TABLE `Dim_LifeStyle` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Dim_Person`
--

DROP TABLE IF EXISTS `Dim_Person`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Dim_Person` (
  `PersonID` int NOT NULL AUTO_INCREMENT,
  `Sex` varchar(10) DEFAULT NULL,
  `AgeGroup` varchar(20) DEFAULT NULL,
  `Education` varchar(50) DEFAULT NULL,
  `Income` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`PersonID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Dim_Person`
--

LOCK TABLES `Dim_Person` WRITE;
/*!40000 ALTER TABLE `Dim_Person` DISABLE KEYS */;
/*!40000 ALTER TABLE `Dim_Person` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Fact_Diabetes`
--

DROP TABLE IF EXISTS `Fact_Diabetes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Fact_Diabetes` (
  `FactID` int NOT NULL AUTO_INCREMENT,
  `PersonID` int DEFAULT NULL,
  `HealthID` int DEFAULT NULL,
  `LifeStyleID` int DEFAULT NULL,
  `AccessID` int DEFAULT NULL,
  `Diabetes_binary` int DEFAULT NULL,
  PRIMARY KEY (`FactID`),
  KEY `PersonID` (`PersonID`),
  KEY `HealthID` (`HealthID`),
  KEY `LifeStyleID` (`LifeStyleID`),
  KEY `AccessID` (`AccessID`),
  CONSTRAINT `Fact_Diabetes_ibfk_1` FOREIGN KEY (`PersonID`) REFERENCES `Dim_Person` (`PersonID`),
  CONSTRAINT `Fact_Diabetes_ibfk_2` FOREIGN KEY (`HealthID`) REFERENCES `Dim_HealthStatus` (`HealthID`),
  CONSTRAINT `Fact_Diabetes_ibfk_3` FOREIGN KEY (`LifeStyleID`) REFERENCES `Dim_LifeStyle` (`LifeStyleID`),
  CONSTRAINT `Fact_Diabetes_ibfk_4` FOREIGN KEY (`AccessID`) REFERENCES `Dim_HealthCareAccess` (`AccessID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Fact_Diabetes`
--

LOCK TABLES `Fact_Diabetes` WRITE;
/*!40000 ALTER TABLE `Fact_Diabetes` DISABLE KEYS */;
/*!40000 ALTER TABLE `Fact_Diabetes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `diabetes_raw`
--

DROP TABLE IF EXISTS `diabetes_raw`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `diabetes_raw` (
  `Diabetes_binary` tinyint DEFAULT NULL,
  `HighBP` tinyint DEFAULT NULL,
  `HighChol` tinyint DEFAULT NULL,
  `CholCheck` tinyint DEFAULT NULL,
  `BMI` float DEFAULT NULL,
  `Smoker` tinyint DEFAULT NULL,
  `Stroke` tinyint DEFAULT NULL,
  `HeartDiseaseorAttack` tinyint DEFAULT NULL,
  `PhysActivity` tinyint DEFAULT NULL,
  `Fruits` tinyint DEFAULT NULL,
  `Veggies` tinyint DEFAULT NULL,
  `HvyAlcoholConsump` tinyint DEFAULT NULL,
  `AnyHealthcare` tinyint DEFAULT NULL,
  `NoDocbcCost` tinyint DEFAULT NULL,
  `GenHlth` tinyint DEFAULT NULL,
  `MentHlth` tinyint DEFAULT NULL,
  `PhysHlth` tinyint DEFAULT NULL,
  `DiffWalk` tinyint DEFAULT NULL,
  `Sex` tinyint DEFAULT NULL,
  `Age` tinyint DEFAULT NULL,
  `Education` tinyint DEFAULT NULL,
  `Income` tinyint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `diabetes_raw`
--

LOCK TABLES `diabetes_raw` WRITE;
/*!40000 ALTER TABLE `diabetes_raw` DISABLE KEYS */;
/*!40000 ALTER TABLE `diabetes_raw` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES (1,'giau','giau@gmail.com','$2b$10$9FHnrtT4A/UQR1reXJ1bFecmfQzQOm6gYzVm3Ibm6LzSifQnGSUMi','2025-12-15 07:40:11'),(2,'rumeodinhau','thien@gmail.com','$2b$10$PI4o/FxchEcxQI7CkjAOR.I4Na54pXSqmFk51iMsxzPw9xeoOd1BS','2025-12-15 07:41:28'),(3,'rua','tuanh@gmail.com','$2b$10$rRnm538jv.58dv2qTlWmVembUEiOMFRnwC50rIaCAzuRG/jBo3KU6','2025-12-16 11:56:28');
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-12-16 21:25:20
