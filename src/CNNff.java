import java.util.Random;

/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNff extends Structure {

    public CNNff(){
    }

    public static network CNNff(network convnet, double[][][] batch_x){

        int n = convnet.layers.size();

        //Random r = new Random(0);
        //randomly dropout some features of the input images
        /*for (int i = 0; i < batch_x.length; i++) {
            for (int j = 0; j < batch_x[0].length; j++) {
                for (int k = 0; k < batch_x[0][0].length; k++) {
                    //Random r = new Random();
                    double randomValue = r.nextDouble();
                    if((double)randomValue<convnet.first_layer_dropout)
                        batch_x[i][j][k]=0;
                }
            }
        }*/

        if(convnet.layers.get(0).a.isEmpty()){
            convnet.layers.get(0).a.add(0,batch_x);
        }else convnet.layers.get(0).a.set(0,batch_x);

        int inputmaps = 1;


        for (int i = 1; i < n; i++) {
            if ("c".equals(convnet.layers.get(i).type)){

                LAYER layer_previous = convnet.layers.get(i-1);
                LAYER layer_current = convnet.layers.get(i);

                if (!layer_current.a.isEmpty()){
                    layer_current.a.clear();
                }

                for (int j = 0; j < layer_current.outmaps; j++) {
                    //temporary output
                    int a = layer_previous.a.get(0).length;
                    int b = layer_previous.a.get(0)[0].length;
                    int c = layer_previous.a.get(0)[0][0].length;
                    int a_new = a - layer_current.kernelsize + 1;
                    int b_new = b - layer_current.kernelsize + 1;
                    int c_new = c;
                    double[][][] z = new double[a_new][b_new][c_new];
                    for (int k = 0; k < a_new; k++) {
                        for (int l = 0; l < b_new; l++) {
                            for (int m = 0; m < c_new; m++) {
                                z[k][l][m] = 0.0;
                            }
                        }
                    }

                    //used maps are convolved and applied ReLU, dropped out maps are assigned values zero
                    //if (layer_current.used_maps[j] == 1) {
                        //convolution
                        /**
                         * each input is convolved with an output-specific kernel and sum of all convolved inputs gives an output
                         */
                        for (int k = 0; k < inputmaps; k++) {
                            //a matrix-wise 2D convolution along the 3rd dimension is used instead of 3D convolution
                            for (int l = 0; l < c_new; l++) {
                                double[][] x_one = new double[a][b];
                                for (int m = 0; m < a; m++) {
                                    for (int o = 0; o < b; o++) {
                                        x_one[m][o] = layer_previous.a.get(k)[m][o][l];
                                    }
                                }
                                double[][] z_conv = Convolution.convolution2D(x_one, a, b, layer_current.k.get(j + k * layer_current.outmaps), layer_current.kernelsize, layer_current.kernelsize);
                                for (int m = 0; m < a_new; m++) {
                                    for (int o = 0; o < b_new; o++) {
                                        z[m][o][l] = z[m][o][l] + z_conv[m][o];
                                    }
                                }
                            }

                        }
                        //apply activation function ReLU
                        double[][][] m = new double[a_new][b_new][c_new];
                        for (int k = 0; k < a_new; k++) {
                            for (int l = 0; l < b_new; l++) {
                                for (int o = 0; o < c_new; o++) {
                                    if (z[k][l][o]+layer_current.b[j]>0.0) {
                                        m[k][l][o] = z[k][l][o] + layer_current.b[j];
                                    } else {
                                        m[k][l][o] = 0.0;
                                    }
                                }
                            }
                        }
                        layer_current.a.add(j,m);
                    //} else {
                    //    layer_current.a.add(j,z);
                    }
               // }
                //number of inputs to the new layer is outmaps of this layer
                inputmaps=layer_current.outmaps;
            }
            else if ("s".equals(convnet.layers.get(i).type)){

                LAYER layer_previous = convnet.layers.get(i-1);
                LAYER layer_current = convnet.layers.get(i);

                if (!layer_current.a.isEmpty()){
                    layer_current.a.clear();
                }

                int a = layer_previous.a.get(0).length;
                int b = layer_previous.a.get(0)[0].length;
                int c = layer_previous.a.get(0)[0][0].length;
                int a_new = a-layer_current.scale+1;
                int b_new = b-layer_current.scale+1;

                double[][] subsample = new double[layer_current.scale][layer_current.scale];
                for (int j = 0; j < layer_current.scale; j++) {
                    for (int k = 0; k < layer_current.scale; k++) {
                        subsample[j][k]=(double) 1/(layer_current.scale*layer_current.scale);
                    }
                }

                //subsampling is carried as convolution with constant kernel
                for (int j = 0; j < inputmaps; j++) {
                    double[][][] z = new double[a_new][b_new][c];
                    //matrix-wise 2D convolution
                    for (int k = 0; k < c; k++) {
                        double[][] a_one = new double[a][b];
                        for (int m = 0; m < a; m++) {
                            for (int o = 0; o < b ; o++) {
                                a_one[m][o] = layer_previous.a.get(j)[m][o][k];
                            }
                        }
                        double[][] a_conv=Convolution.convolution2D(a_one,a,b,subsample,layer_current.scale,layer_current.scale);
                        for (int l = 0; l < a_new; l++) {
                            for (int m = 0; m < b_new; m++) {
                                z[l][m][k] = a_conv[l][m];
                            }
                        }
                    }
                    //get only one row and column in each two, resulting in reducing the size to its half
                    double[][][] m = new double[a/layer_current.scale][b/layer_current.scale][c];
                    for (int k = 0; k < a/layer_current.scale; k++) {
                        for (int l = 0; l < b/layer_current.scale; l++) {
                            for (int o = 0; o < c; o++) {
                                m[k][l][o]=z[k*2][l*2][o];
                            }
                        }
                    }
                    layer_current.a.add(j,m);
                }
            }
        }

        /**
         * concatenate all end layer feature maps into vector
         */
        LAYER layer_current = convnet.layers.get(n-1);
        int a = layer_current.a.get(0).length;
        int b = layer_current.a.get(0)[0].length;
        int c = layer_current.a.get(0)[0][0].length;
        int outputmaps = layer_current.a.size();
        convnet.fv= new double[a*b*outputmaps][c];
        for (int i = 0; i < c; i++) {
            int row = 0;
            for (int j = 0; j < outputmaps; j++) {
                for (int k = 0; k < b; k++) {
                    for (int l = 0; l < a; l++) {
                        convnet.fv[row][i]=layer_current.a.get(j)[l][k][i];
                        row++;
                    }
                }
            }
        }


        //feedforward into output perceptrons
        //ffw*fv
        int d = convnet.ffW.length;
        int e = convnet.fv.length;
        double[][] product = new double[d][c];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < c; j++) {
                product[i][j]=0;
            }
        }
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < c; j++) {
                for (int k = 0; k < e; k++) {
                    product[i][j]=product[i][j]+convnet.ffW[i][k]*convnet.fv[k][j];
                }
            }
        }

        //apply activation function ReLU
        convnet.o = new double[d][c];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < c; j++) {
                if(product[i][j]>0.0)
                    convnet.o[i][j]=product[i][j];
                else convnet.o[i][j]=0.0;
            }
        }

        return convnet;
    }
}
