pipeline {
  agent {
    docker {
      image 'guyzsarun/hls4ml:vivado'
    }

  }
  stages {
    stage('qkeras-api-test') {
      parallel {
        stage('python-api-test') {
          steps {
            sh './run-test test_qkeras_api.py'
          }
        }

        stage('vivado-test') {
          steps {
            sh './run-test test_qkeras_vivado.py'
          }
        }
      }
      stage('keras-api-test'){
        {
          parallel{
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
  options {
    timeout(time: 3, unit: 'HOURS')
  }
}
