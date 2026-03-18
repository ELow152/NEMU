#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <GLFW/glfw3.h>
#include "loader.h"
#include <inttypes.h>

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768

/* Adjust how many synapse lines to emit per frame. Increase for fewer frames. */
#ifndef MAX_LINES_PER_FRAME
#define MAX_LINES_PER_FRAME 200000   /* safe default for CPU-only rendering */
#endif

static float rotX = 0.0f, rotY = 0.0f;
static float zoom = 1.0f;
static double lastMouseX = -1, lastMouseY = -1;
static int dragging = 0;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(action == GLFW_PRESS || action == GLFW_REPEAT){
        switch(key){
            case GLFW_KEY_W: zoom *= 1.1f; break;
            case GLFW_KEY_S: zoom /= 1.1f; break;
            case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, 1); break;
        }
    }
}

void mouse_button_callback(GLFWwindow* window,int button,int action,int mods){
    if(button == GLFW_MOUSE_BUTTON_LEFT){
        if(action == GLFW_PRESS){
            dragging = 1;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        } else if(action == GLFW_RELEASE){
            dragging = 0;
        }
    }
}

void cursor_position_callback(GLFWwindow* window,double xpos,double ypos){
    if(dragging){
        rotY += (float)(xpos - lastMouseX) * 0.5f;
        rotX += (float)(ypos - lastMouseY) * 0.5f;
        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

void set_projection(int width,int height){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (float)width / (float)height;
    glOrtho(-aspect, aspect, -1, 1, -10, 10);
    glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char **argv){
    const char *path = "serialized.nemu";
    if (argc > 1) path = argv[1];

    Brain brain = load_nemu(path);
    printf("Loaded %zu neurons\n", brain.n_neurons);

    if(!glfwInit()){ fprintf(stderr,"GLFW init failed\n"); return -1; }
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Fly Brain Viewer", NULL, NULL);
    if(!window){ glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window,key_callback);
    glfwSetMouseButtonCallback(window,mouse_button_callback);
    glfwSetCursorPosCallback(window,cursor_position_callback);

    glEnable(GL_POINT_SMOOTH);
    glPointSize(2.0f);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    /* Compute bounding box including synapses */
    float minX=1e30f, minY=1e30f, minZ=1e30f;
    float maxX=-1e30f, maxY=-1e30f, maxZ=-1e30f;
    uint64_t total_synapses = 0;

    for(size_t i=0;i<brain.n_neurons;i++){
        Neuron* n = &brain.neurons[i];
        if(n->x<minX) minX=n->x; if(n->y<minY) minY=n->y; if(n->z<minZ) minZ=n->z;
        if(n->x>maxX) maxX=n->x; if(n->y>maxY) maxY=n->y; if(n->z>maxZ) maxZ=n->z;

        total_synapses += n->n_out_synapses;
        for(uint64_t j=0;j<n->n_out_synapses;j++){
            Synapse* s = &n->out_synapses[j];
            if(s->pre_x<minX) minX=s->pre_x; if(s->pre_y<minY) minY=s->pre_y; if(s->pre_z<minZ) minZ=s->pre_z;
            if(s->pre_x>maxX) maxX=s->pre_x; if(s->pre_y>maxY) maxY=s->pre_y; if(s->pre_z>maxZ) maxZ=s->pre_z;
            if(s->post_x<minX) minX=s->post_x; if(s->post_y<minY) minY=s->post_y; if(s->post_z<minZ) minZ=s->post_z;
            if(s->post_x>maxX) maxX=s->post_x; if(s->post_y>maxY) maxY=s->post_y; if(s->post_z>maxZ) maxZ=s->post_z;
        }
    }

    float scaleX = maxX - minX, scaleY = maxY - minY, scaleZ = maxZ - minZ;
    float maxScale = scaleX; if(scaleY>maxScale) maxScale=scaleY; if(scaleZ>maxScale) maxScale=scaleZ;
    if (maxScale <= 0.0f) maxScale = 1.0f;
    float centerX = (minX+maxX)/2.0f, centerY = (minY+maxY)/2.0f, centerZ = (minZ+maxZ)/2.0f;

    printf("BBox: min(%.3f,%.3f,%.3f) max(%.3f,%.3f,%.3f) total_synapses=%" PRIu64 "\n",
           minX,minY,minZ, maxX,maxY,maxZ, total_synapses);

    /* progressive drawing state for synapse lines */
    size_t draw_neuron_idx = 0;
    uint64_t draw_synapse_idx = 0; /* index within current neuron */
    uint64_t drawn_synapses = 0;
    int first_frame_done = 0;

    /* Pre-calc point color per type_id could be cached if many types; keep simple here */
    while(!glfwWindowShouldClose(window)){
        int width, height;
        glfwGetFramebufferSize(window,&width,&height);
        glViewport(0,0,width,height);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        set_projection(width,height);
        glLoadIdentity();
        glScalef(zoom / maxScale, zoom / maxScale, zoom / maxScale);
        glTranslatef(-centerX,-centerY,-centerZ);
        glRotatef(rotX,1.0f,0.0f,0.0f);
        glRotatef(rotY,0.0f,1.0f,0.0f);

        /* Draw neurons */
        glBegin(GL_POINTS);
        for(size_t i=0;i<brain.n_neurons;i++){
            Neuron* n = &brain.neurons[i];
            float r = (n->type_id % 256) / 255.0f;
            float g = ((n->type_id * 2) % 256) / 255.0f;
            float b = ((n->type_id * 3) % 256) / 255.0f;
            glColor3f(r,g,b);
            glVertex3f(n->x,n->y,n->z);
        }
        glEnd();

        /* Draw synapses progressively to avoid single-frame stall.
           Use a single color for all synapses per batch (less overhead). */
        uint64_t budget = MAX_LINES_PER_FRAME;
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_LINES);
        while(budget > 0 && draw_neuron_idx < brain.n_neurons){
            Neuron* pre = &brain.neurons[draw_neuron_idx];
            if (pre->n_out_synapses == 0) {
                draw_neuron_idx++;
                draw_synapse_idx = 0;
                continue;
            }
            while(draw_synapse_idx < pre->n_out_synapses && budget > 0){
                Synapse* s = &pre->out_synapses[draw_synapse_idx];
                glVertex3f(s->pre_x, s->pre_y, s->pre_z);
                glVertex3f(s->post_x, s->post_y, s->post_z);
                draw_synapse_idx++;
                drawn_synapses++;
                budget--;
            }
            if(draw_synapse_idx >= pre->n_out_synapses){
                draw_neuron_idx++;
                draw_synapse_idx = 0;
            }
        }
        glEnd();

        /* If we've drawn all synapses once, keep them visible but optionally stop incrementing. */
        if(drawn_synapses >= total_synapses){
            /* do nothing: all lines have been submitted at least once */
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        if(!first_frame_done){ printf("Fly brain rendered (progressive). total_synapses=%" PRIu64 "\n", total_synapses); first_frame_done=1; }
    }

    free_brain(&brain);
    glfwTerminate();
    return 0;
}