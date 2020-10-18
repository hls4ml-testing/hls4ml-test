pipeline {
  agent {
    docker {
      image 'guyzsarun/hls4ml:vivado'
    }
  }
  options {
    timeout(time: 3, unit: 'HOURS')
  }
  
  stages {
    stage('qkeras-api-test') {
      steps {
        sh 'pytest test_qkeras_api.py'
      }
    }

  }
}
