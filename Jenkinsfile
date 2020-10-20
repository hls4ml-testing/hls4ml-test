pipeline {
  agent {
    docker {
      image 'guyzsarun/hls4ml:vivado'
    }

  }
  stages {
    stage('qkeras-api-test') {
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

  }
  options {
    timeout(time: 3, unit: 'HOURS')
  }
}
