/**
 * Created by ziyihua on 17/07/15.
 */
public class CNNbp extends Structure {

    public CNNbp(){
    }

    public static network CNNbp(network convnet, int[][] batch_y){
        int n = convnet.layers.size();

        //error
        convnet.e=new double[convnet.o.length][convnet.o[0].length];
        for (int i = 0; i < convnet.o.length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                convnet.e[i][j]=convnet.o[i][j]-(double)batch_y[i][j];
            }
        }

        //loss function
        double sum=0;
        for (int i = 0; i < convnet.o.length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                sum=sum+convnet.e[i][j]*convnet.e[i][j];
            }
        }
        double loss;
        loss = 0.5 * sum /convnet.o[0].length;
        convnet.L=loss;

        /**
         * backpropagation delta
         */
        //output delta od=e.*(o>0)
        convnet.od = new double[convnet.o.length][convnet.o[0].length];
        for (int i = 0; i < convnet.o.length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                if(convnet.o[i][j]>0.0) {
                    convnet.od[i][j] = convnet.e[i][j];
                }
                else {
                    convnet.od[i][j]=0.0;
                }
            }
        }



        //feature vector delta fvd=ffw'*od
        convnet.fvd = new double[convnet.ffW[0].length][convnet.o[0].length];
        for (int i = 0; i < convnet.ffW[0].length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                convnet.fvd[i][j] = 0;
            }
        }
        for (int i = 0; i < convnet.ffW[0].length; i++) {
            for (int j = 0; j < convnet.o[0].length; j++) {
                for (int k = 0; k < convnet.o.length; k++) {
                    convnet.fvd[i][j]=convnet.fvd[i][j]+convnet.ffW[k][i]*convnet.od[k][j];
                }
            }
        }

        //only convolution layer has sigmoid function and if the last layer is convolution
        //fvd=fvd.*[fv.*(1-fv)]
        if ("c".equals(convnet.layers.get(n-1).type)){
            double[][] dotprod_2 = new double[convnet.fvd.length][convnet.fvd[0].length];
            for (int i = 0; i < convnet.fvd.length; i++) {
                for (int j = 0; j < convnet.fvd[0].length; j++) {
                    dotprod_2[i][j]=convnet.fv[i][j]*(1-convnet.fv[i][j]);
                }
            }
            for (int i = 0; i < convnet.fvd.length; i++) {
                for (int j = 0; j < convnet.fvd[0].length; j++) {
                    convnet.fvd[i][j]=convnet.fvd[i][j]*dotprod_2[i][j];
                }
            }
        }

        /**
         * reshape feature vector deltas into output map style
         */
        LAYER layer_last = convnet.layers.get(n - 1);

        layer_last.d.clear();

        int a = layer_last.a.get(0).length;
        int b = layer_last.a.get(0)[0].length;
        int c = layer_last.a.get(0)[0][0].length;
        int outputmaps = layer_last.a.size();


        for (int i = 0; i < outputmaps; i++) {
            double[][][] d = new double[a][b][c];
            for (int j = 0; j < c; j++) {
                for (int k = 0; k < b; k++) {
                    for (int l = 0; l < a; l++) {
                        d[l][k][j]=convnet.fvd[i*a*b+k*a+l][j];
                    }
                }
            }
            layer_last.d.add(i,d);
        }

        for (int i = n-2; i >=0 ; i--) {
            if("c".equals(convnet.layers.get(i).type)){

                if (!convnet.layers.get(i).d.isEmpty()){
                    convnet.layers.get(i).d.clear();
                }

                int num_a = convnet.layers.get(i).a.size();
                int size_a1=convnet.layers.get(i).a.get(0).length;
                int size_a2=convnet.layers.get(i).a.get(0)[0].length;
                int size_a3=convnet.layers.get(i).a.get(0)[0][0].length;

                //assigned to unused maps
                double[][][] zeros = new double[size_a1][size_a2][size_a3];
                for (int k = 0; k < size_a1; k++) {
                    for (int l = 0; l < size_a2; l++) {
                        for (int m = 0; m < size_a3; m++) {
                            zeros[k][l][m]=0.0;
                        }
                    }
                }

                for (int j = 0; j < num_a; j++) {
                    //if (convnet.layers.get(i).used_maps[j]==1) {
                        //expand the size of feature deltas to that of the current layer (replicate each row and column n times where n is the scale of the next subsampling layer)
                        int sub_scale = convnet.layers.get(i + 1).scale;
                        double[][][] expand = new double[size_a1][size_a2][size_a3];
                        for (int k = 0; k < size_a3; k++) {
                            for (int l = 0; l < size_a2; l++) {
                                for (int m = 0; m < size_a1; m++) {
                                    expand[m][l][k] = convnet.layers.get(i + 1).d.get(j)[m / sub_scale][l / sub_scale][k];
                                }
                            }
                        }
                        //d_current_layer=ReLU'(a).*expand(d_previous_layer)/scale^2)
                        double[][][] derivative = new double[size_a1][size_a2][size_a3];
                        for (int k = 0; k < size_a3; k++) {
                            for (int l = 0; l < size_a2; l++) {
                                for (int m = 0; m < size_a1; m++) {
                                    if (convnet.layers.get(i).a.get(j)[m][l][k]>0) {
                                        derivative[m][l][k] = 1.0;
                                    }
                                    else {
                                        derivative[m][l][k]=0.0;
                                    }
                                }
                            }
                        }

                        double[][][] product_1 = new double[size_a1][size_a2][size_a3];
                        for (int k = 0; k < size_a3; k++) {
                            for (int l = 0; l < size_a2; l++) {
                                for (int m = 0; m < size_a1; m++) {
                                    product_1[m][l][k] = derivative[m][l][k] * expand[m][l][k]/((double) sub_scale * sub_scale);
                                }
                            }
                        }
                        convnet.layers.get(i).d.add(j,product_1);
                    //}
                    //else {
                    //    convnet.layers.get(i).d.add(j,zeros);
                    //}
                }
            }else if ("s".equals(convnet.layers.get(i).type)){

                if (!convnet.layers.get(i).d.isEmpty()){
                    convnet.layers.get(i).d.clear();
                }

                int num_a = convnet.layers.get(i).a.size();
                int size_a1=convnet.layers.get(i).a.get(0).length;
                int size_a2=convnet.layers.get(i).a.get(0)[0].length;
                int size_a3=convnet.layers.get(i).a.get(0)[0][0].length;

                for (int j = 0; j < num_a; j++){

                    double[][][] z = new double[size_a1][size_a2][size_a3];
                    for (int k = 0; k < size_a1; k++) {
                        for (int l = 0; l < size_a2; l++) {
                            for (int m = 0; m < size_a3; m++) {
                                z[k][l][m]=0;
                            }
                        }
                    }
                    //reverse-convolution
                    for (int k = 0; k < convnet.layers.get(i+1).a.size(); k++) {
                        //matrix-wise convolution along 3rd dimension
                        for (int l = 0; l < size_a3; l++) {
                            int size_d1 = convnet.layers.get(i+1).d.get(j).length;
                            int size_d2 = convnet.layers.get(i+1).d.get(j)[0].length;
                            double[][] d_one = new double[size_d1][size_d2];
                            for (int m = 0; m < size_d2; m++) {
                                for (int o = 0; o < size_d1; o++) {
                                    d_one[o][m] = convnet.layers.get(i+1).d.get(k)[o][m][l];
                                }
                            }
                            double[][] z_conv;
                            z_conv = Convolution.convolution2D_full_wof(d_one, size_d1, size_d2, convnet.layers.get(i + 1).k.get(k + j * convnet.layers.get(i+1).outmaps),convnet.layers.get(i+1).kernelsize,convnet.layers.get(i+1).kernelsize);
                            for (int m = 0; m < size_a2; m++) {
                                for (int o = 0; o < size_a1; o++) {
                                    z[o][m][l]=z[o][m][l]+z_conv[o][m];
                                }
                            }
                        }
                    }convnet.layers.get(i).d.add(j,z);
                }
            }
        }

        /**
         * Calculate gradients
         */
        for (int i = 1; i < n; i++) {
            if("c".equals(convnet.layers.get(i).type)) {

                convnet.layers.get(i).dk.clear();

                int num_a_current = convnet.layers.get(i).a.size();
                int num_a_previous = convnet.layers.get(i - 1).a.size();

                for (int k = 0; k < num_a_previous; k++) {
                    for (int j = 0; j < num_a_current; j++) {
                        //gradient dk is calculated as convolution of a in the last layer with k in the current layer
                        double[][][] flipall = flip3D(convnet.layers.get(i - 1).a.get(k));
                        double[][][] kernel = convnet.layers.get(i).d.get(j);
                        double[][] conv = Convolution.convolution3D(flipall, flipall.length, flipall[0].length, flipall[0][0].length, kernel, kernel.length, kernel[0].length, kernel[0][0].length);
                        double[][] dk = new double[conv.length][conv[0].length];
                        for (int l = 0; l < conv.length; l++) {
                            for (int m = 0; m < conv[0].length; m++) {
                                dk[l][m] = conv[l][m] / (double) kernel[0][0].length;
                            }
                        }
                        convnet.layers.get(i).dk.add(j+k*convnet.layers.get(i).outmaps,dk);
                    }
                }

                //db=sum(d)/size
                convnet.layers.get(i).db = new double[num_a_current];
                for (int j = 0; j < num_a_current; j++) {
                    double d_sum = 0;
                    int size_d1 = convnet.layers.get(i).d.get(j).length;
                    int size_d2 = convnet.layers.get(i).d.get(j)[0].length;
                    int size_d3 = convnet.layers.get(i).d.get(j)[0][0].length;
                    for (int k = 0; k < size_d1; k++) {
                        for (int l = 0; l < size_d2; l++) {
                            for (int m = 0; m < size_d3; m++) {
                                d_sum = d_sum + convnet.layers.get(i).d.get(j)[k][l][m];
                            }
                        }
                    }
                    convnet.layers.get(i).db[j] = d_sum / (double) size_d3;
                }
            }
        }

        //dffW=od*fv'/size
        convnet.dffW = new double[convnet.od.length][convnet.fv.length];
        for (int i = 0; i < convnet.od.length; i++) {
            for (int j = 0; j < convnet.fv.length; j++) {
                convnet.dffW[i][j]=0;
            }
        }

        for (int row = 0; row < convnet.dffW.length; row++) {
            for (int i = 0; i < convnet.dffW[0].length; i++) {
                for (int j = 0; j < convnet.fv[0].length; j++) {
                    convnet.dffW[row][i]=convnet.dffW[row][i]+convnet.od[row][j]*convnet.fv[i][j];
                }
            }
        }

        for (int i = 0; i < convnet.dffW.length; i++) {
            for (int j = 0; j < convnet.dffW[0].length; j++) {
                convnet.dffW[i][j]=convnet.dffW[i][j]/(double) convnet.od[0].length;
            }
        }


        //dffb=0
        convnet.dffb = new double[convnet.od.length];
        for (int i = 0; i < convnet.od.length; i++) {
            convnet.dffb[i]=0;
        }


        return convnet;
    }
}
