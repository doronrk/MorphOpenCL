#include "ofApp.h"

#include "MSAOpenCL.h"

//Particle type - contains all information about particle except particle's position.
typedef struct{
	float4 target;  //target point where to fly
	float speed;    //speed of flying
	float dummy1;
	float dummy2;
	float dummy3;
} Particle;

/*
  Dummy fields are needed to comply OpenCL alignment rule:
  sizeof(float4) = 4*4=16,
  sizeof(float) = 4,
  so overall structure size should divide to 16 and 4.
  Without dummies the size if sizeof(float4)+sizeof(float)=20, so we add
  three dummies to have size 32 bytes.
 */

msa::OpenCL			opencl;

msa::OpenCLBufferManagedT<Particle>	particles; // vector of Particles on host and corresponding clBuffer on device
msa::OpenCLBufferManagedT<float4> particlePos; // vector of particle positions on host and corresponding clBuffer on device
GLuint vbo;

int N = 1000000; //Number of particles
//int N = 10000;

//--------------------------------------------------------------
void ofApp::setup(){
    //Screen setup
    ofSetWindowTitle("Morph OpenCL example");
    ofSetFrameRate( 60 );
	ofSetVerticalSync(false);
    signalParticleSpeed = 0.10;
    spectrumParticleSpeed = 0.10;
    faceParticleSpeed = 0.04;
    cubeParticleSpeed = 0.04;
    
    portionOfSpecToDraw = 1.0;
    freqScalingExponent = 1.7;
    amplitudeScalingExponent = 1.6;
    amplitudeScale = 500.0;
    if(portionOfSpecToDraw > 1.0 || portionOfSpecToDraw <= 0.0) {
        ofLogFatalError() << "invalid portionOfSpecToDraw, should be in range (0, 1.0] ";
    }
    
    // Audio setup
    soundStream.listDevices();
    soundStream.setDeviceID(0);
    nOutputChannels = 0;
    nInputChannels = 2;
    sampleRate = 44100;
    bufferSize = 2048;
    nBuffers = 4;
    fft = ofxFft::create(bufferSize, OF_FFT_WINDOW_HAMMING, OF_FFT_FFTW);
    drawFFTBins.resize(fft->getBinSize() * portionOfSpecToDraw);
    middleFFTBins.resize(fft->getBinSize());
    audioFFTBins.resize(fft->getBinSize());
    drawSignal.resize(bufferSize);
    middleSignal.resize(bufferSize);
    audioSignal.resize(bufferSize);
    
    soundStream.setup(this, nOutputChannels, nInputChannels, sampleRate, bufferSize, nBuffers);
    
    //Camera
	cam.setDistance(800);
    cam.setFov(60.0);
//    cam.disableMouseInput();
    cam.enableMouseInput();
    
    //OpenCL
	opencl.setupFromOpenGL();
	opencl.loadProgramFromFile("Particle.cl");
	opencl.loadKernel("updateParticle");
    
    //create vbo which holds particles positions - particlePos, for drawing
    glGenBuffersARB(1, &vbo);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(float4) * N, 0, GL_DYNAMIC_COPY_ARB);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    
    // init host and CL buffers
    particles.initBuffer( N );
    particlePos.initFromGLObject( vbo, N );
    
    // load the face
    const char* imageName = "ksenia.jpg";
    loadImage(imageName);
    
    faceWave = false;
    suspended = false;
    drawMode = FACE;
}

void ofApp::loadImage(const char* name)
{
    ofPixels pix;
    ofLoadImage(pix, name);
    faceWidth = pix.getWidth();
    faceHeight = pix.getHeight();
    
    //Build "distribution array" of brightness
    int sum = 0;
    for (int y=0; y<faceHeight; y++) {
        for (int x=0; x<faceWidth; x++) {
            sum += pix.getColor(x, y).getBrightness();
        }
    }
    facePoints.resize(sum);
    
    int q = 0;
    for (int y=0; y<faceHeight; y++) {
        for (int x=0; x<faceWidth; x++) {
            int v = pix.getColor(x, y).getBrightness();
            for (int i=0; i<v; i++) {
                facePoints[q++] = ofPoint( x, y );
            }
        }
    }
}


//--------------------------------------------------------------
void ofApp::update(){
    //Update particles positions
    
    //Link parameters to OpenCL (see Particle.cl):
    opencl.kernel("updateParticle")->setArg(0, particles.getCLMem());
	opencl.kernel("updateParticle")->setArg(1, particlePos.getCLMem());
   
    //Execute OpenCL computation and wait it finishes
    opencl.kernel("updateParticle")->run1D( N );
	opencl.finish();
    
    // update bin sizes
    soundMutex.lock();
    for (int i = 0; i < drawFFTBins.size(); i++)
    {
        drawFFTBins[i] = abs(middleFFTBins[i] * (1 + pow(i, freqScalingExponent)/drawFFTBins.size()));
        drawFFTBins[i] = pow(drawFFTBins[i], amplitudeScalingExponent);
    }
    drawSignal = middleSignal;
    soundMutex.unlock();
    normalize(drawFFTBins);
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(0, 0, 0);

    //camera rotate
    float time = ofGetElapsedTimef();
//    cam.orbit( sin(time*0.25) * 6, 0, 600, ofPoint( 0, 0, 0 ) );
    cam.begin();
    
    //Enabling "addition" blending mode to sum up particles brightnesses
    ofEnableBlendMode( OF_BLENDMODE_ADD );
    
    ofSetColor( 16, 16, 16 );
    glPointSize(1.0);
    
    //Drawing particles
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof( float4 ), 0);
	glDrawArrays(GL_POINTS, 0, N );
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    
    
    switch(drawMode)
    {
        case CUBE  : morphToCube(true);   break;
        case FACE  :
//            morphToFace(facePoints, faceWidth, faceHeight);
            break;
        case SPECTRUM  : morphToSpectrum(drawFFTBins);   break;
        case SIGNAL  : morphToSignal(drawSignal);   break;
    }
    ofEnableAlphaBlending();    //Restore from "addition" blending mode
    
    cam.end();
    
    ofSetColor( ofColor::white );
//    ofDrawBitmapString( "1 - morph to cube, 2 - morph to face", 20, 20 );
    
}


//--------------------------------------------------------------
void ofApp::morphToSignal(vector<float> signal)
{
    float signalWidth = 600;
    int nSamples = signal.size();
    float sampleWidth = signalWidth / nSamples;
    
    if (nSamples > 0)
    {
        for (int i = 0; i < N; i++)
        {
            int sampleNumber = i % nSamples;
            float px = ((sampleNumber * sampleWidth) * 2) - signalWidth;
            float py = signal[sampleNumber] * amplitudeScale;
            float pz = 0.0;
            
            Particle& p = particles[i];
            p.target.set(px, py, pz, 0.0);
            p.speed = signalParticleSpeed;
        }
    }
    
    particles.writeToDevice();
}

//--------------------------------------------------------------
void ofApp::morphToSpectrum(vector<float> bins)
{
    float spectrumWidth = 600;
    int nBins = bins.size();
    float binWidth = spectrumWidth / nBins;

    if (nBins > 0)
    {
        for (int i = 0; i < N; i++)
        {
            int binNumber = i % nBins;
            float px = ((binNumber * binWidth) * 2) - spectrumWidth;
            float py = bins[binNumber] * amplitudeScale; // magnitude of the bin
            float pz = 0.0;
            
            //Setting to particle
            Particle &p = particles[i];
            p.target.set(px, py, pz, 0.0);
            p.speed = spectrumParticleSpeed;
        }
    }
    
    //upload to GPU
    particles.writeToDevice();
}



//--------------------------------------------------------------
void ofApp::morphToCube( bool setPos ) {       //Morphing to cube
	for(int i=0; i<N; i++) {
		//Getting random point at cube
        float rad = 90;
        ofPoint pnt( ofRandom(-rad, rad), ofRandom(-rad, rad), ofRandom(-rad, rad) );
        
        //project point on cube's surface
        int axe = ofRandom( 0, 3 );
        if ( axe == 0 ) { pnt.x = ( pnt.x >= 0 ) ? rad : (-rad ); }
        if ( axe == 1 ) { pnt.y = ( pnt.y >= 0 ) ? rad : (-rad ); }
        if ( axe == 2 ) { pnt.z = ( pnt.z >= 0 ) ? rad : (-rad ); }
        axe = (axe + 1)%3;
        if ( axe == 0 ) { pnt.x = ( pnt.x >= 0 ) ? rad : (-rad ); }
        if ( axe == 1 ) { pnt.y = ( pnt.y >= 0 ) ? rad : (-rad ); }
        if ( axe == 2 ) { pnt.z = ( pnt.z >= 0 ) ? rad : (-rad ); }
        
        //add noise
//        float noise = 10;
//        pnt.x += ofRandom( -noise, noise );
//        pnt.y += ofRandom( -noise, noise );
//        pnt.z += ofRandom( -noise, noise );
//        
//        pnt.y -= 150;   //shift down

        //Setting to particle
		Particle &p = particles[i];
        p.target.set( pnt.x, pnt.y, pnt.z, 0 );
        p.speed = cubeParticleSpeed;
        
        if ( setPos ) {
            particlePos[i].set( pnt.x, pnt.y, pnt.z, 0 );
        }
	}
    
    //upload to GPU
    particles.writeToDevice();
    if ( setPos ) {
        particlePos.writeToDevice();
    }
}

void ofApp::doFaceWave()
{
    float time = ofGetElapsedTimef();
    float height = sin(time) * 300;
    ofPushStyle();
        ofSetColor(245, 58, 135);
        ofSetLineWidth(2.0);
        ofNoFill();
        float lineWidth = 600;
        ofBeginShape();
        ofVertex(-lineWidth, height, 0.0);
        ofVertex(lineWidth, height, 0.0);
        ofEndShape(false);
    ofPopStyle();
}


//--------------------------------------------------------------
void ofApp::morphToFace(vector<ofPoint> facePoints, int w, int h) {      //Morphing to face
    //All drawn particles have equal brightness, so to achieve face-like
    //particles configuration by placing different number of particles
    //at each pixel and draw them in "addition blending" mode.
    
    //Loading image
    //(Currently we recalculate distribution each time
    //- so try to use diferent images for morph, selected randomly)
//    ofPixels pix;
//    ofLoadImage(pix, "ksenia.jpg");
//    int w = pix.getWidth();
//    int h = pix.getHeight();
//
//    //Build "distribution array" of brightness
//    int sum = 0;
//    for (int y=0; y<h; y++) {
//        for (int x=0; x<w; x++) {
//            sum += pix.getColor(x, y).getBrightness();
//        }
//    }
//    vector<ofPoint> tPnt( sum );
//    
//    int q = 0;
//    for (int y=0; y<h; y++) {
//        for (int x=0; x<w; x++) {
//            int v = pix.getColor(x, y).getBrightness();
//            for (int i=0; i<v; i++) {
//                tPnt[q++] = ofPoint( x, y );
//            }
//        }
//    }
    
    //Set up particles
    float scl = 2;
    float noisex = 2.5;
    float noisey = 0.5;
    float noisez = 5.0;

    int sum = facePoints.size();
    cerr << "sum: " << sum << endl;
    for(int i=0; i<N; i++) {
		Particle &p = particles[i];
        
//        int q = ofRandom( 0, sum );
        int q = i % sum;
        ofPoint pnt = facePoints[q];
        pnt.x -= w/2;   //centering
        pnt.y -= h/2;
        pnt.x *= scl;       //scaling
        pnt.y *= -scl;
        
//        add noise to x, y
        pnt.x += ofRandom( -scl/2, scl/2 );
        pnt.y += ofRandom( -scl/2, scl/2 );
        
        pnt.x += ofRandom( -noisex, noisex );
        pnt.y += ofRandom( -noisey, noisey );
        
        //projection on cylinder
        float Rad = w * scl * 0.4;
        pnt.z = sqrt( fabs( Rad * Rad - pnt.x * pnt.x ) ) - Rad;
        
        //add noise to z
        pnt.z += ofRandom( -noisez, noisez );
        
        
        //set to particle
        p.target.set( pnt.x, pnt.y, pnt.z, 0 );
        p.speed = faceParticleSpeed;

    }
    
//    if (faceWave)
//    {
//        doFaceWave();
//    }
    
    //upload to GPU
    particles.writeToDevice();
    particlePos.writeToDevice();
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if ( key == '1' )
    {
        drawMode = CUBE;
    }
    else if ( key == '2' )
    {
        drawMode = FACE;
        morphToFace(facePoints, faceWidth, faceHeight);
    }
    else if ( key ==  '3' )
    {
        drawMode = SPECTRUM;
    }
    else if ( key ==  '4' )
    {
        drawMode = SIGNAL;
    }
    else if ( key == '9')
    {
        faceWave = ! faceWave;
    }
}

//--------------------------------------------------------------
void ofApp::audioReceived(float* input, int bufferSize, int nChannels)
{
    vector<float> monoMix;
    monoMix.resize(bufferSize);
    // scale buffer by maxValue
    bool allZero = true;
    for (int frame = 0; frame < bufferSize; frame++)
    {
        float sum = 0.0;
        for (int channel = 0; channel < nChannels; channel++)
        {
            int i = frame*nChannels + channel;
            if (input[i] != 0.0) {
                allZero = false;
            }
            sum = sum + input[i];
        }
        sum = sum / nChannels;
        monoMix[frame] = sum;
    }
    if (allZero) {
        suspended = true;
    } else if (suspended)
    {
        suspended = false;
    }
    normalize(monoMix);
    //calculate the fft
    fft->setSignal(monoMix);
    float* curFft = fft->getAmplitude();
    memcpy(&audioFFTBins[0], curFft, sizeof(float) * fft->getBinSize());
    memcpy(&audioSignal[0], input, sizeof(float) * bufferSize);
    normalize(audioFFTBins);
    
    soundMutex.lock();
    middleFFTBins = audioFFTBins;
    middleSignal = audioSignal;
    soundMutex.unlock();
}

//--------------------------------------------------------------
void ofApp::normalize(vector<float>& data) {
    float maxValue = 0;
    for(int i = 0; i < data.size(); i++) {
        if(abs(data[i]) > maxValue) {
            maxValue = abs(data[i]);
        }
    }
    if (maxValue > 0)
    {
        for(int i = 0; i < data.size(); i++) {
            data[i] /= maxValue;
        }
    }
}


//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    
}
