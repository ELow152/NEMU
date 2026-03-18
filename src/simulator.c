// fly_brain_simulator.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <GLFW/glfw3.h>
#include "loader.h"

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768
#define SIM_STEP 0.01f  // simulation timestep in seconds

typedef struct SimNeuron {
    Neuron base;            // original neuron data
    float membrane_potential;
    int fired;
    float threshold;
    float decay;
    float refractory_time;
} SimNeuron;

typedef struct SimBrain {
    SimNeuron* neurons;
    size_t n_neurons;
} SimBrain;

// ---------------- Simulation Logic ----------------
void init_sim_brain(SimBrain* sim, Brain* brain) {
    sim->n_neurons = brain->n_neurons;
    sim->neurons = malloc(sizeof(SimNeuron)*brain->n_neurons);
    for(size_t i=0;i<brain->n_neurons;i++){
        sim->neurons[i].base = brain->neurons[i];
        sim->neurons[i].membrane_potential = 0.0f;
        sim->neurons[i].fired = 0;
        sim->neurons[i].threshold = 1.0f;
        sim->neurons[i].decay = 0.95f;
        sim->neurons[i].refractory_time = 0.0f;
    }
}

void simulate_step(SimBrain* sim, float dt) {
    for(size_t i=0;i<sim->n_neurons;i++){
        SimNeuron* n = &sim->neurons[i];

        // Refractory handling
        if(n->refractory_time > 0.0f){
            n->refractory_time -= dt;
            n->fired = 0;
            continue;
        }

        // Decay membrane potential
        n->membrane_potential *= n->decay;

        // Sum inputs from firing neighbors
        for(size_t j=0;j<n->base.n_neighbors;j++){
            uint64_t target_id = n->base.neighbors[j];
            for(size_t k=0;k<sim->n_neurons;k++){
                if(sim->neurons[k].base.root_id == target_id && sim->neurons[k].fired){
                    n->membrane_potential += 0.5f;  // synaptic weight
                }
            }
        }

        // Check threshold
        if(n->membrane_potential >= n->threshold){
            n->fired = 1;
            n->membrane_potential = 0.0f;
            n->refractory_time = 0.02f; // 20ms refractory
        } else {
            n->fired = 0;
        }
    }
}

// ---------------- Visualization Helpers ----------------
static float rotX=0.0f, rotY=0.0f;
static double lastMouseX=-1,lastMouseY=-1;
static int dragging=0;

void key_callback(GLFWwindow* window,int key,int scancode,int action,int mods){
    if(action==GLFW_PRESS||action==GLFW_REPEAT){
        if(key==GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window,1);
    }
}
void mouse_button_callback(GLFWwindow* window,int button,int action,int mods){
    if(button==GLFW_MOUSE_BUTTON_LEFT){
        if(action==GLFW_PRESS){ dragging=1; glfwGetCursorPos(window,&lastMouseX,&lastMouseY);}
        else if(action==GLFW_RELEASE) dragging=0;
    }
}
void cursor_position_callback(GLFWwindow* window,double xpos,double ypos){
    if(dragging){ rotY += (xpos-lastMouseX)*0.5f; rotX += (ypos-lastMouseY)*0.5f; lastMouseX=xpos; lastMouseY=ypos;}
}

void set_projection(int width,int height){
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    float aspect=(float)width/(float)height;
    glOrtho(-aspect,aspect,-1,1,-10,10);
    glMatrixMode(GL_MODELVIEW);
}

void render_brain(SimBrain* sim){
    // Compute bounds & center
    float minX=1e9,minY=1e9,minZ=1e9,maxX=-1e9,maxY=-1e9,maxZ=-1e9;
    for(size_t i=0;i<sim->n_neurons;i++){
        Neuron* n=&sim->neurons[i].base;
        if(n->x<minX) minX=n->x; if(n->y<minY) minY=n->y; if(n->z<minZ) minZ=n->z;
        if(n->x>maxX) maxX=n->x; if(n->y>maxY) maxY=n->y; if(n->z>maxZ) maxZ=n->z;
    }
    float scaleX=maxX-minX,scaleY=maxY-minY,scaleZ=maxZ-minZ;
    float maxScale=(scaleX>scaleY?scaleX:scaleY); if(scaleZ>maxScale) maxScale=scaleZ;
    float centerX=(minX+maxX)/2.0f,centerY=(minY+maxY)/2.0f,centerZ=(minZ+maxZ)/2.0f;

    glLoadIdentity();
    glScalef(1.0f/maxScale,1.0f/maxScale,1.0f/maxScale);
    glTranslatef(-centerX,-centerY,-centerZ);
    glRotatef(rotX,1.0f,0.0f,0.0f);
    glRotatef(rotY,0.0f,1.0f,0.0f);

    // Neurons
    glBegin(GL_POINTS);
    for(size_t i=0;i<sim->n_neurons;i++){
        SimNeuron* n=&sim->neurons[i];
        if(n->fired) glColor3f(1.0f,0.0f,0.0f);
        else{
            float r=(n->base.type_id%256)/255.0f;
            float g=((n->base.type_id*2)%256)/255.0f;
            float b=((n->base.type_id*3)%256)/255.0f;
            glColor3f(r,g,b);
        }
        glVertex3f(n->base.x,n->base.y,n->base.z);
    }
    glEnd();

    // Synapses
    glColor3f(1.0f,0.0f,0.0f);
    glBegin(GL_LINES);
    for(size_t i=0;i<sim->n_neurons;i++){
        SimNeuron* n=&sim->neurons[i];
        if(!n->fired) continue;
        for(size_t j=0;j<n->base.n_neighbors;j++){
            uint64_t pid=n->base.neighbors[j];
            for(size_t k=0;k<sim->n_neurons;k++){
                if(sim->neurons[k].base.root_id==pid){
                    glVertex3f(n->base.x,n->base.y,n->base.z);
                    glVertex3f(sim->neurons[k].base.x,sim->neurons[k].base.y,sim->neurons[k].base.z);
                    break;
                }
            }
        }
    }
    glEnd();
}

// ---------------- Main for testing (commented out) ----------------
/*
int main(){
    Brain brain = load_nemu("serialized.nemu");
    printf("Loaded %zu neurons\n",brain.n_neurons);

    SimBrain sim;
    init_sim_brain(&sim,&brain);

    if(!glfwInit()){ fprintf(stderr,"GLFW init failed\n"); return -1; }
    GLFWwindow* window=glfwCreateWindow(WINDOW_WIDTH,WINDOW_HEIGHT,"Fly Brain Simulator",NULL,NULL);
    if(!window){ glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window,key_callback);
    glfwSetMouseButtonCallback(window,mouse_button_callback);
    glfwSetCursorPosCallback(window,cursor_position_callback);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(2.0f);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f,0.1f,0.1f,1.0f);

    while(!glfwWindowShouldClose(window)){
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        int width,height;
        glfwGetFramebufferSize(window,&width,&height);
        glViewport(0,0,width,height);
        set_projection(width,height);

        simulate_step(&sim,SIM_STEP);
        render_brain(&sim);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    for(size_t i=0;i<brain.n_neurons;i++) free(brain.neurons[i].neighbors);
    free(brain.neurons);
    free(sim.neurons);
    return 0;
}
*/