/**
 * Created by ziyihua on 17/07/15.
 */
public class Convolution extends Structure {

    public Convolution(){
    }

    public static double singlePixelConvolution(double [][] input,
                                                int x, int y,
                                                double [][] k,
                                                int kernelWidth,
                                                int kernelHeight){

        double output = 0;
        for(int i=0;i<kernelWidth;++i){
            for(int j=0;j<kernelHeight;++j){
                output = output + (input[x+i][y+j] * k[i][j]);
            }
        }
        return output;
    }

    public static double singlePixelConvolution3D(double [][][] input,
                                                  int x, int y,
                                                  double [][][] k,
                                                  int kernelWidth,
                                                  int kernelHeight,
                                                  int kernelDepth){
        double output = 0;
        for(int i=0;i<kernelWidth;++i){
            for(int j=0;j<kernelHeight;++j){
                for (int l = 0; l < kernelDepth; l++) {
                    output = output + (input[x + i][y + j][l] * k[i][j][l]);
                }
            }
        }
        return output;
    }

    public static double [][] convolution2D(double [][] input,
                                            int width, int height,
                                            double [][] kernel,
                                            int kernelWidth,
                                            int kernelHeight){

        /**
         * flip kernel
         */
        double[][] kernel_flipped = new double[kernelWidth][kernelHeight];
        for (int i = 0; i < kernelWidth; i++) {
            for (int j = 0; j < kernelHeight; j++) {
                kernel_flipped[i][j]=kernel[kernelWidth-1-i][kernelHeight-1-j];
            }
        }


        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        double [][] output = new double [smallWidth][smallHeight];
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


    public static double [][] convolution2D_full_wof(double [][] input,
                                                     int width, int height,
                                                     double [][] kernel,
                                                     int kernelWidth,
                                                     int kernelHeight){

        /**
         * transform input
         */
        double[][] input_new = new double[width+2*kernelWidth-2][height+2*kernelHeight-2];
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
        double[][] output = new double[smallWidth][smallHeight];
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


    public static double[][] convolution3D(double[][][] input, int width, int height, int depth, double[][][] kernel, int kernelWidth, int kernelHeight, int kernelDepth){
        /**
         * input and kernel has the same third dimension
         */
        double[][] output = new double[width-kernelWidth+1][height-kernelHeight+1];
        double[][][] kernel_flipped = flip3D(kernel);

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
