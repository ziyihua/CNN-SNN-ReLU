/**
 * Created by ziyihua on 17/07/15.
 */
public class Convolution extends Structure {

    public Convolution(){
    }

    public static float singlePixelConvolution(float [][] input,
                                                int x, int y,
                                                float [][] k,
                                                int kernelWidth,
                                                int kernelHeight){

        float output = 0;
        for(int i=0;i<kernelWidth;++i){
            for(int j=0;j<kernelHeight;++j){
                output = output + (input[x+i][y+j] * k[i][j]);
            }
        }
        return output;
    }

    public static float singlePixelConvolution3D(float [][][] input,
                                                  int x, int y,
                                                  float [][][] k,
                                                  int kernelWidth,
                                                  int kernelHeight,
                                                  int kernelDepth){
        float output = 0;
        for(int i=0;i<kernelWidth;++i){
            for(int j=0;j<kernelHeight;++j){
                for (int l = 0; l < kernelDepth; l++) {
                    output = output + (input[x + i][y + j][l] * k[i][j][l]);
                }
            }
        }
        return output;
    }

    public static float [][] convolution2D(float [][] input,
                                            int width, int height,
                                            float [][] kernel,
                                            int kernelWidth,
                                            int kernelHeight){

        /**
         * flip kernel
         */
        float[][] kernel_flipped = new float[kernelWidth][kernelHeight];
        for (int i = 0; i < kernelWidth; i++) {
            for (int j = 0; j < kernelHeight; j++) {
                kernel_flipped[i][j]=kernel[kernelWidth-1-i][kernelHeight-1-j];
            }
        }


        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        float [][] output = new float [smallWidth][smallHeight];
        for(int i=0;i<smallWidth;++i){
            for(int j=0;j<smallHeight;++j){
                output[i][j]=0;
            }
        }
        for(int i=0;i<smallWidth;++i){
            for(int j=0;j<smallHeight;++j){
                output[i][j] = singlePixelConvolution(input,i,j,kernel_flipped,
                        kernelWidth,kernelHeight);
            }
        }
        return output;
    }


    public static float [][] convolution2D_full_wof(float [][] input,
                                                     int width, int height,
                                                     float [][] kernel,
                                                     int kernelWidth,
                                                     int kernelHeight){

        /**
         * transform input
         */
        float[][] input_new = new float[width+2*kernelWidth-2][height+2*kernelHeight-2];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                input_new[i][j]=0;
            }
        }
        for (int i = kernelWidth-1; i < width+kernelWidth-1; i++) {
            for (int j = kernelHeight-1; j < height+kernelHeight-1; j++) {
                input_new[i][j]=input[i-kernelWidth+1][j-kernelHeight+1];
            }
        }

        width=width+2*kernelWidth-2;
        height=height+2*kernelHeight-2;


        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        float[][] output = new float[smallWidth][smallHeight];
        for(int i=0;i<smallWidth;++i){
            for(int j=0;j<smallHeight;++j){
                output[i][j]=0;
            }
        }
        for(int i=0;i<smallWidth;++i){
            for(int j=0;j<smallHeight;++j){
                output[i][j] = singlePixelConvolution(input_new,i,j,kernel,
                        kernelWidth,kernelHeight);
            }
        }
        return output;
    }


    public static float[][] convolution3D(float[][][] input, int width, int height, int depth, float[][][] kernel, int kernelWidth, int kernelHeight, int kernelDepth){
        /**
         * input and kernel has the same third dimension
         */
        float[][] output = new float[width-kernelWidth+1][height-kernelHeight+1];
        float[][][] kernel_flipped = flip3D(kernel);

        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        for(int i=0;i<smallWidth;++i){
            for(int j=0;j<smallHeight;++j){
                output[i][j]=0;
            }
        }
        for(int i=0;i<smallWidth;++i){
            for(int j=0;j<smallHeight;++j){
                output[i][j] = singlePixelConvolution3D(input,i,j,kernel_flipped,
                        kernelWidth,kernelHeight,kernelDepth);
            }
        }

        return output;
    }
}
