pipeline {
  agent {
    docker {
      image 'guyzsarun/hls4ml:vivado'
    }

  }
  stages {
    stage('qkeras-api-test') {
      steps {
        sh 'pytest test_qkeras_api.py'
      }
    }

  }
}