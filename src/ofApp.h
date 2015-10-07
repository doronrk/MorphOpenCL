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

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    
    void morphToCube( bool setPos );            //Morphing to cube
    void morphToFace();                         //Morphing to face
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
    ofSoundStream soundStream;
    
    int nOutputChannels;
    int nInputChannels;
    int sampleRate;
    int bufferSize;
    int nBuffers;
    
    float portionOfSpecToDraw;
    float freqScalingExponent;
    float amplitudeScalingExponent;
    float amplitudeScale;
    
    float signalParticleSpeed;
    float spectrumParticleSpeed;
    float faceParticleSpeed;
    float cubeParticleSpeed;

    
    bool drawingSpectrum;
    bool suspended;
    bool drawingSignal;
    
    void normalize(vector<float>& data);

};
