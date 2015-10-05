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

//--------------------------------------------------------------
void ofApp::setup(){
    //Screen setup
    ofSetWindowTitle("Morph OpenCL example");
    ofSetFrameRate( 60 );
	ofSetVerticalSync(false);
    particleSpeed = 0.05;
    
    // Audio setup
    soundStream.listDevices();
    soundStream.setDeviceID(3);
    nOutputChannels = 0;
    nInputChannels = 2;
    sampleRate = 44100;
    bufferSize = 4096;
    nBuffers = 4;
    magnitudeScale = 200.0;
    fft = ofxFft::create(bufferSize, OF_FFT_WINDOW_HAMMING, OF_FFT_FFTW);
    drawBins.resize(fft->getBinSize());
    middleBins.resize(fft->getBinSize());
    audioBins.resize(fft->getBinSize());
    soundStream.setup(this, nOutputChannels, nInputChannels, sampleRate, bufferSize, nBuffers);
    
    //Camera
	cam.setDistance(600);
    cam.setFov(60.0);
//    cam.disableMouseInput();
    
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
    
//    drawingSpectrum = true;
    morphToFace();
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
//    soundMutex.lock();
    drawBins = middleBins;
//    soundMutex.unlock();
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(0, 0, 0);

    //camera rotate
//    float time = ofGetElapsedTimef();
//    cam.orbit( sin(time*0.5) * 12, 0, 600, ofPoint( 0, 0, 0 ) );
    cam.begin();
    
    //Enabling "addition" blending mode to sum up particles brightnesses
    ofEnableBlendMode( OF_BLENDMODE_ADD );
    
    ofSetColor( 16, 16, 16 );
    glPointSize( 1 );
    
    //Drawing particles
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof( float4 ), 0);
	glDrawArrays(GL_POINTS, 0, N );
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    
    if (drawingSpectrum)
    {
        morphToSpectrum(drawBins);
    }
    
    ofEnableAlphaBlending();    //Restore from "addition" blending mode
    
    cam.end();
    
    ofSetColor( ofColor::white );
    ofDrawBitmapString( "1 - morph to cube, 2 - morph to face", 20, 20 );
    
}

void ofApp::drawReferenceSphere(float px, float py, float pz)
{
    ofPushStyle();
    ofSetColor(245, 58, 135);
    ofSetLineWidth(2.0);
    
    ofSpherePrimitive sphere;
    sphere.setRadius(90);
    sphere.setResolution(10);
    sphere.setMode(OF_PRIMITIVE_LINES);
    sphere.setPosition(0, 0, 0);
    sphere.drawWireframe();
    
    ofPopStyle();
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
            float py = magnitudeScale * sqrt(bins[binNumber]); // magnitude of the bin
            float pz = 0.0;
            
            //Setting to particle
            Particle &p = particles[i];
            p.target.set(px, py, pz, 0);
            p.speed = particleSpeed;
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
        p.speed = particleSpeed;
        
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


//--------------------------------------------------------------
void ofApp::morphToFace() {      //Morphing to face
    //All drawn particles have equal brightness, so to achieve face-like
    //particles configuration by placing different number of particles
    //at each pixel and draw them in "addition blending" mode.
    
    //Loading image
    //(Currently we recalculate distribution each time
    //- so try to use diferent images for morph, selected randomly)
    ofPixels pix;
    ofLoadImage(pix, "ksenia.jpg");
    int w = pix.getWidth();
    int h = pix.getHeight();

    //Build "distribution array" of brightness
    int sum = 0;
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            sum += pix.getColor(x, y).getBrightness();
        }
    }
    vector<ofPoint> tPnt( sum );
    
    int q = 0;
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v = pix.getColor(x, y).getBrightness();
            for (int i=0; i<v; i++) {
                tPnt[q++] = ofPoint( x, y );
            }
        }
    }
    
    //Set up particles
    float scl = 2;
    float noisex = 2.5;
    float noisey = 0.5;
    float noisez = 5.0;

    for(int i=0; i<N; i++) {
		Particle &p = particles[i];
        
        //целевая точка
        int q = ofRandom( 0, sum );
        ofPoint pnt = tPnt[ q ];
        pnt.x -= w/2;   //centering
        pnt.y -= h/2;
        pnt.x *= scl;       //scaling
        pnt.y *= -scl;
        
        //add noise to x, y
        pnt.x += ofRandom( -scl/2, scl/2 );
        pnt.y += ofRandom( -scl/2, scl/2 );
        
        pnt.x += ofRandom( -noisex, noisex );
        pnt.y += ofRandom( -noisey, noisey );
        
        //peojection on cylinder
        float Rad = w * scl * 0.4;
        pnt.z = sqrt( fabs( Rad * Rad - pnt.x * pnt.x ) ) - Rad;
        
        //add noise to z
        pnt.z += ofRandom( -noisez, noisez );
        
        
        //set to particle
        p.target.set( pnt.x, pnt.y, pnt.z, 0 );
        p.speed = 0.06;
//        p.speed = 1.0;

        
    }
    
    //upload to GPU
    particles.writeToDevice();
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if ( key == '1' )
    {
        morphToCube( false );
        drawingSpectrum = false;
    }
    if ( key == '2' )
    {
        morphToFace();
        drawingSpectrum = false;
    }
    if ( key ==  '3' )
    {
        drawingSpectrum = ! drawingSpectrum;
    }
}

//--------------------------------------------------------------
void ofApp::audioReceived(float* input, int bufferSize, int nChannels)
{
    // calculate the largest value in the buffer
    float maxValue = 0.0;
    for (int i = 0; i < bufferSize * nInputChannels; i++)
    {
        if (abs(input[i]) > maxValue)
        {
            maxValue = abs(input[i]);
        }
    }
    float monoMix[bufferSize];
    // scale buffer by maxValue
    for (int frame = 0; frame < bufferSize; frame++)
    {
        float sum = 0.0;
        for (int channel = 0; channel < nChannels; channel++)
        {
            int i = frame*nChannels + channel;
            float normalizedAmp = input[i];
            if (maxValue > 0.0)
            {
                normalizedAmp = normalizedAmp / maxValue;
            }
            sum = sum + normalizedAmp;
        }
        sum = sum / nChannels;
        monoMix[frame] = sum;
    }
    //calculate the fft
    fft->setSignal(monoMix);
    float* curFft = fft->getAmplitude();
    memcpy(&audioBins[0], curFft, sizeof(float) * fft->getBinSize());
    
    //scale the bins
    maxValue = 0;
    for (int i = 0; i < fft->getBinSize(); i++)
    {
        if (abs(audioBins[i]) > maxValue)
        {
            maxValue = abs(audioBins[i]);
        }
    }
    for (int i = 0; i < fft->getBinSize(); i++)
    {
        if (maxValue > 0.0)
        {
            audioBins[i] = audioBins[i] / maxValue;
        }
    }
    
//    soundMutex.lock();
    middleBins = audioBins;
//    soundMutex.unlock();
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
