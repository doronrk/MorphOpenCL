#include "ofApp.h"


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
    soundStream.setDeviceID(4);
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
    const char* imageName = "ksenia256.jpg";
    loadImage(imageName);
    downsampledBins.resize(faceMatrix.size(), 0);
//    downsampledBins.resize(8, 0);
    
    yFaceWave = 0;
    yFaceWaveDelta = 1;
    
    faceWave = false;
    suspended = false;
    drawMode = FACE;
    
//    static const int testSourceArr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//    static const int testTargetArr[] = {0, 0, 0, 0, 0};
//
//    vector<float> testSource (testSourceArr, testSourceArr + sizeof(testSourceArr) / sizeof(testSourceArr[0]) );
//    vector<float> testTarget (testTargetArr, testTargetArr + sizeof(testTargetArr) / sizeof(testTargetArr[0]) );
//    
//    downsampleBins(testTarget, testSource);
}

void ofApp::loadImage(const char* name)
{
    ofPixels pix;
    ofLoadImage(pix, name);
    
    int height = pix.getHeight();
    int width = pix.getWidth();
    
    //now we have an empty 2D-matrix of size (0,0). Resizing it with one single command:
    faceMatrix.resize(height, vector<float>(width ,0.0));
    
    //Build "distribution array" of brightness
    float sumBrightness = 0;
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            float brightness = pix.getColor(x, y).getBrightness();
            faceMatrix[y][x] = brightness;
            sumBrightness += brightness;
        }
    }
    for (int y=0; y< height; y++) {
        for (int x=0; x< width; x++) {
            faceMatrix[y][x] = faceMatrix[y][x] / sumBrightness;
        }
    }
}

//--------------------------------------------------------------
void ofApp::downsampleBins(vector<float>& target, vector<float>& source)
{
    int nTargetBins = target.size();
    if (nTargetBins < 1)
    {
        cerr << "nTargetBins < 1" << endl;
        return;
    }
    int nSourceBins = source.size();
    if (nTargetBins > nSourceBins)
    {
        cerr << "nTargetBins > nSourceBins" << endl;
        return;
    }
    int ratio = nSourceBins / nTargetBins;
    for (int targetBin = 0; targetBin < nTargetBins; targetBin++)
    {
        float sum = 0;
        for (int sourceBin = 0; sourceBin < ratio; sourceBin++)
        {
            int i = targetBin*ratio + sourceBin;
            sum += source[i];
        }
        float average = sum / (float) ratio;
        target[targetBin] = average;
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
    downsampleBins(downsampledBins, drawFFTBins);
//    normalize(drawFFTBins);
    normalize(downsampledBins);
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
//            morphToFace(faceMatrix);
//            doFaceSplit();
//            downsampleBins(downsampledBins, drawFFTBins);
            doFaceSpectrum(downsampledBins);
            break;
        case SPECTRUM  :
//            morphToSpectrum(drawFFTBins);
//            downsampleBins(downsampledBins, drawFFTBins);
            morphToSpectrum(downsampledBins);
            break;
        case SIGNAL  :
            morphToSignal(drawSignal);
            break;
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
            ofApp::Particle &p = particles[i];
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

void ofApp::doFaceSpectrum(vector<float> bins)
{
    if (bins.size() != faceParticles.size())
    {
        cerr << "bins.size() != faceParticles.size(), in doFaceSpectrum" << endl;
        cerr << "bins.size(): " << bins.size() << endl;
        cerr << "faceParticles.size(): " << faceParticles.size() << endl;
        return;
    }
    for (int y = 0; y < faceParticles.size(); y++)
    {
        float magnitude = bins[y];
        for (int x = 0; x < faceParticles[0].size(); x++)
        {
            vector<Particle*> particlesAtPixel = faceParticles[y][x];
            for (int p = 0; p < particlesAtPixel.size(); p++)
            {
                Particle* part = particlesAtPixel[p];
                part->target.set(part->target.x - magnitude, part->target.y, part->target.z, 0);
            }
        }
    }
    particles.writeToDevice();
}


//-----------------------------------------------------------------------------------------
void ofApp::doFaceSplit()
{
    float time = ofGetElapsedTimef();
    int faceHeight = faceParticles.size();
    float height = sin(time) * faceHeight;
    ofPushStyle();
        ofSetColor(245, 58, 135);
        ofSetLineWidth(2.0);
        ofNoFill();
        float lineWidth = 600;
        ofBeginShape();
        ofVertex(-lineWidth, height, 0.0);
        ofVertex(lineWidth, height, 0.0);
        ofEndShape(false);
    
        if (faceHeight > 0)
        {
            int y = ((height) + faceHeight) / 2;
            cerr << "y: " << y << endl;
            vector<vector<Particle*> > particleRow = faceParticles[y];
            for (int x = 0; x < particleRow.size(); x++)
            {
                vector<Particle*> particlesAtPixel = particleRow[x];
                for (int p = 0; p < particlesAtPixel.size(); p++)
                {
                    Particle* part = particlesAtPixel[p];
                    part->target.set(part->target.x - 200, part->target.y, part->target.z, 0);
                }
            }
        }
    ofPopStyle();
    particles.writeToDevice();
}


//--------------------------------------------------------------
void ofApp::morphToFace(vector< vector<float> > faceMatrix) {      //Morphing to face
    
    //Set up particles
    float noisex = 2.5;
    float noisey = 0.5;
    float noisez = 5.0;
    
    int height = faceMatrix.size();
    if (height < 1) {return;}
    int width = faceMatrix[0].size();
    
    faceParticles.clear();
    faceParticles.resize(height, vector<vector<Particle*> >(width, vector<Particle*>()));

    
    int particleIndex = 0;
    float px, py, pz;
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            float brightness = faceMatrix[y][x];
            int nParticles = brightness * N;
            for (int z = 0; z < nParticles; z++)
            {
                Particle &p = particles[particleIndex++];
                px = (x - width/2) * faceScale;
                py = (y - height/2) * -faceScale;
                px += ofRandom( -faceScale/2, faceScale/2 );
                py += ofRandom( -faceScale/2, faceScale/2 );
                
                px += ofRandom( -noisex, noisex );
                py += ofRandom( -noisey, noisey );
                
                //projection on cylinder
                float Rad = width * faceScale * 0.4;
                pz = sqrt( fabs( Rad * Rad - px * px ) ) - Rad;
                
                //add noise to z
                pz += ofRandom( -noisez, noisez );
                //set to particle
                p.target.set(px, py, pz, 0);
                faceParticles[y][x].push_back(&p);
                p.speed = faceParticleSpeed;
            }
        }
    }
    
    // assign remainder particles
    for (; particleIndex < N; particleIndex++)
    {
        int x = ofRandom(0, width);
        int y = ofRandom(0, height);
        Particle &p = particles[particleIndex];
        px = (x - width/2) * faceScale;
        py = (y - height/2) * -faceScale;
        px += ofRandom( -faceScale/2, faceScale/2 );
        py += ofRandom( -faceScale/2, faceScale/2 );
        
        px += ofRandom( -noisex, noisex );
        py += ofRandom( -noisey, noisey );
        
        //projection on cylinder
        float Rad = width * faceScale * 0.4;
        pz = sqrt( fabs( Rad * Rad - px * px ) ) - Rad;
        
        //add noise to z
        pz += ofRandom( -noisez, noisez );
        //set to particle
        p.target.set(px, py, pz, 0);
        faceParticles[y][x].push_back(&p);
        p.speed = faceParticleSpeed;
        
    }
    
    //upload to GPU
    particles.writeToDevice();
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
        morphToFace(faceMatrix);
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
