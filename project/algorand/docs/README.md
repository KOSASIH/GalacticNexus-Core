# Galactic Nexus Core
======================

[![Build Status](https://img.shields.io/github/workflow/status/KOSASIH/GalacticNexus-Core/CI?label=Build&logo=github)](https://github.com/KOSASIH/GalacticNexus-Core/actions)
[![License](https://img.shields.io/github/license/KOSASIH/GalacticNexus-Core?label=License&logo=github)](https://github.com/KOSASIH/GalacticNexus-Core/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Online-blue?logo=readthedocs)](https://galacticnexus-core.readthedocs.io/en/latest/)

## Table of Contents

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [System Requirements](#system-requirements)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [API Documentation](#api-documentation)
* [Contributing](#contributing)
* [License](#license)

## Introduction
---------------

Galactic Nexus Core is a cutting-edge, open-source project that enables seamless interactions between various blockchain networks and the Algorand ecosystem. This project aims to provide a robust, scalable, and secure framework for developers to build innovative applications.

## Getting Started
-----------------

### Prerequisites

* Node.js (>= 14.17.0)
* npm (>= 6.14.13)
* Docker (>= 20.10.0)
* Docker Compose (>= 1.29.2)

### Installation

1. Clone the repository: `git clone https://github.com/KOSASIH/GalacticNexus-Core.git`
2. Navigate to the project directory: `cd GalacticNexus-Core`
3. Install dependencies: `npm install`
4. Start the development environment: `docker-compose up -d`

### Configuration

* Create a `.env` file in the project root with the following variables:
	+ `ALGORAND_NODE_URL`: Algorand node URL
	+ `ALGORAND_INDEXER_URL`: Algorand indexer URL
	+ `ALGORAND_ACCOUNT`: Algorand account address
	+ `ALGORAND_PRIVATE_KEY`: Algorand private key

### Usage

* Start the application: `npm start`
* Access the API documentation: `http://localhost:3000/docs`

## API Documentation
-------------------

The API documentation is available at [http://localhost:3000/docs](http://localhost:3000/docs).

## Contributing
------------

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to the project.

## License
-------

Galactic Nexus Core is licensed under the [Apache License 2.0](LICENSE).
