#pragma once

//---------------------------------------------------------------
//Example of using OpenCL for creating particle system,
//which morphs between two 3D shapes - cube and face image
//
//Control keys: 1 - morph to cube, 2 - morph to face
//
//All drawn particles have equal brightness, so to achieve face-like
//particles configuration by placing different number of particles
//at each pixel and draw them in "addition blending" mode.
//
//Project is developed for openFrameworks 8.4_osx and is based
//on example-Particles example of ofxMSAOpenCL adoon.
//It uses addons ofxMSAOpenCL and ofxMSAPingPong.
//For simplicity this addons are placed right in the project's folder.
//
//The code and "ksenia.jpg" photo made by Kuflex.com, 2014:
//Denis Perevalov, Igor Sodazot and Ksenia Lyashenko.
//---------------------------------------------------------------


#include "ofMain.h"
#include "ofxFft.h"
#include "MSAOpenCL.h"
class ofApp : public ofBaseApp{
    

public:
    enum DrawMode {TIME, FREQUENCY, FACE_FREQUENCY, MELT, SPLIT};
    
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

    
    /*
     Dummy fields are needed to comply OpenCL alignment rule:
     sizeof(float4) = 4*4=16,
     sizeof(float) = 4,
     so overall structure size should divide to 16 and 4.
     Without dummies the size if sizeof(float4)+sizeof(float)=20, so we add
     three dummies to have size 32 bytes.
     */

    void setup();
    void update();
    void draw();
    
    void morphToCube( bool setPos );            //Morphing to cube
    void morphToFace(vector< vector<float> > faceMatrix);                         //Morphing to face
    void doFaceSpectrum(vector<float> bins);
    void doFaceSplit();
    void doFaceMelt(vector<float> bins, int direction);
    void morphToSpectrum(vector<float> bins);   //Morphing to frequency spectrum
    void morphToSignal(vector<float> bins);     //Morphing to frequency signal

    
    void drawSpectrum(vector<float> bins);      // draw the frequency spectrum as of shape

    
    ofEasyCam cam; // add mouse controls for camera movement

    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    void audioReceived(float* input, int bufferSize, int nChannels);

private:
    ofxFft* fft;
    ofMutex soundMutex;
    vector<float> drawFFTBins, middleFFTBins, audioFFTBins, drawSignal, middleSignal, audioSignal;
    vector<float> downsampledBins;

    vector< vector<float> > faceMatrix;
    vector< vector<vector<ofApp::Particle*> > > faceParticles;
    ofSoundStream soundStream;
    
    int nOutputChannels;
    int nInputChannels;
    int sampleRate;
    int bufferSize;
    int nBuffers;
    
    float faceScale = 3;

    
    float portionOfSpecToDraw;
    float freqScalingExponent;
    float amplitudeScalingExponent;
    float signalAmplitudeScale;
    float spectrumAmplitudeScale;
    
    float signalParticleSpeed;
    float spectrumParticleSpeed;
    float faceParticleSpeed;
    float cubeParticleSpeed;
    
    int yFaceWave;
    int yFaceWaveDelta;
    
    int ySpectrumVerticalShift;
    float ignoreFFTbelow;

    DrawMode drawMode;
    bool suspended;
    bool faceWave;
    
    bool instructionsHidden;

    void normalize(vector<float>& data);
    void cutoff(vector<float>& data, float cutoff, float ignoreBelow);
    void loadImage(const char* name);
    void downsampleBins(vector<float>& target, vector<float>& source);
};
