## Descrição

Este repositório é destinado para versionamento de código de um experimento que visa analisar bibliotecas de deep learning em cenários de Internet of Things (IoT). Para isso, foram elaboradas diversas situações de aplicações envolvendo o uso de Deep learning em um cenários com recursos computacionais limitado (Neste experimento foi utilizado uma Raspberry Pi 3B), cada aplicação foi implementada com bibliotecas diferentes, ao final da execução foram analisadas: a eficiência e uso de recursos de cada biblioteca.

Na atual etapa da proposta, foram analisadas as seguintes bibliotecas:

 - [TensorFlow v 2.0](https://www.tensorflow.org/): Uma plataforma de código aberto de ponta a ponta para aprendizado de máquina. Possui um ecossistema abrangente e flexível de ferramentas, bibliotecas e recursos da comunidade que permite que os pesquisadores desenvolvam o que há de mais moderno em ML e os desenvolvedores construam e implantem facilmente aplicativos desenvolvidos por ML (Descrição oficial).
 - [Pytorch v 1.0](https://pytorch.org/): PyTorch is an optimized tensor library for deep learning using GPUs and CPUs (Descrição oficial).
 - Futuramente outras bibliotecas serão avaliadas.


## Preparando o Ambiente

Neste experimento, foi utilizado uma Raspberry Pi 3, este tipo de computador é constantemente utilizado em aplicações IoT devido a quantidade de recursos computacionais em um dispositivo de tamanho aproximado de uma carteira, além de um baixo custo quando comparado a um computador convencional. Porém, também é necessário considerar as limitações de hardware que tal dispositivo apresenta, o que pode representar desafios para implementação de soluções mais complexas.

O Raspbian foi o sistema operacional utilizado, trata-se de uma versão resumida do Lunix Debian. A interface gráfica deste sistema oferece uma alternativa agradável para os responsáveis pelo desenvolvimento da aplicação.

#### Instalação das Bibliotecas

 - Instruções para instalação do TensorFlow [Aqui](https://www.tensorflow.org/install/source_rpi)
 - Instruções para instalação do Pytorch [Aqui](https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531)
 - Instruções para instaçã odo Pytorch V4 (ainda não testado) [Aqui](https://medium.com/hardware-interfacing/how-to-install-pytorch-v4-0-on-raspberry-pi-3b-odroids-and-other-arm-based-devices-91d62f2933c7)

## Descrição dos Experimentos

Para este experimento, foram utilizados diversas situações aplicáveis em cenários IoT reais.

### Application 1: Classificação de Numeração de Residências

Descrição:

Banco utilizado:

Configuração genérica da redes neural:

### Application 2: Classificação de Consumo de Energia

Descrição:

Banco utilizado:

Configuração genérica da redes neural:

### Application 3: Predição de Rota de Carros

Descrição:

Banco utilizado:

Configuração genérica da redes neural:

## Implementação das Aplicações