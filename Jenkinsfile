pipeline {
  agent {
    docker {
      image 'guyzsarun/hls4ml:vivado'
    }

  }
  stages {
    stage('qkeras') {
      parallel {
        stage('qkeras-api-test') {
          steps {
            sh './run-test test_qkeras_api.py'
          }
        }

        stage('qkeras-vivado-test') {
          steps {
            sh './run-test test_qkeras_vivado.py'
          }
        }

      }
    }

    stage('keras') {
      parallel {
        stage('keras-api-test') {
          steps {
            sh './run-test test_keras_api.py'
          }
        }

        stage('keras-vivado-test') {
          steps {
            sh './run-test test_keras_vivado.py'
          }
        }

      }
    }

  }
}