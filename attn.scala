import scala.io.Source
import spatial.dsl._

//takes in CSV, NO COMMA SEPARATION, separate line for each element
//total # of lines = total # of elements

@spatial class LoadCSV extends SpatialTest {
    type T = FixPt[TRUE,_16,_8] // Fixed point notation, [signed, 16 bits, 8 bits fraction]


    override def runtimeArgs = "0"
    def main(args: Array[String]): Unit = {
        val N = 1
        val sequence_length = 302
        val L = 302
        val L_q = 100
        val E = 512
        val n_heads = 8

        val q_values = loadCSV1D[T]("/Users/marcoandonosie/EE109/finalproject/ee109_final_project/data_dump/model.transformer.decoder.layers.6.multihead_attn/q.csv", "\n")
        val weight_q_values = loadCSV1D[T]("/Users/marcoandonosie/EE109/finalproject/ee109_final_project/data_dump/model.transformer.decoder.layers.6.multihead_attn/weight_q.csv", "\n")
        val bias_q_values = loadCSV1D[T]("/Users/marcoandonosie/EE109/finalproject/ee109_final_project/data_dump/model.transformer.decoder.layers.6.multihead_attn/bias_q.csv", "\n")
        val q_proj_values = loadCSV1D[T]("/Users/marcoandonosie/EE109/finalproject/ee109_final_project/data_dump/model.transformer.decoder.layers.6.multihead_attn/q_proj.csv", "\n")
        val q_output_values = Array.tabulate(L_q * N * E){ i => 0.to[T] } //zeroed out values

        // DRAMs for inputs and outputs
        val q = DRAM[T](L_q, N, E)
        val weight_q = DRAM[T](E, E)
        val bias_q = DRAM[T](E, N) //512 x 1
        val q_proj = DRAM[T](L_q, N, E) //golden to reference q_output against
        val q_output = DRAM[T](L_q, N, E)


        /*val K = DRAM[Float](L, N, E)
        val V = DRAM[Float](L, N, E)
        val Wq = DRAM[Float](E, E)
        val Wk = DRAM[Float](E, E)
        val Wv = DRAM[Float](E, E)
        val Wo = DRAM[Float](E, E)
        val Out = DRAM[Float](L, N, E)*/


        //setMem step 
        setMem(q, q_values)
        setMem(weight_q, weight_q_values)
        setMem(bias_q, bias_q_values)
        setMem(q_proj, q_proj_values)
        setMem(q_output, q_output_values)

        
        Accel {
            val q_sram = SRAM[T](L_q, N, E)
            val weight_q_sram = SRAM[T](E, E)
            val bias_q_sram = SRAM[T](E, N)
            val q_proj_sram = SRAM[T](L_q, N, E)

             // Load input matrices into SRAM
            q_sram load q
            weight_q_sram load weight_q
            bias_q_sram load bias_q

            // Compute q_proj = q × weight_q.T + bias_q. 
            //TODO: make the following foreach loop NOT take 5 mins to run
           Foreach(L_q by 1, N by 1, E by 1) { (l, n, e_out) =>
                val sum = Reg[T](0.to[T])
                Foreach(E by 1) { e_in =>
                sum := sum + q_sram(l, n, e_in) * weight_q_sram(e_in, e_out)
                }
                q_proj_sram(l, n, e_out) = sum + bias_q_sram(e_out, n)
            }

            q_output store q_proj_sram
        }


        //test things out 
        println("N = " + N)
        println("L = " + L)
        println("E = " + E)
        println("q values: " + q_values(1).to[T])

        //test that q_output is created at all; print one specific element, see if any output is created
        val q_output_host = getMem(q_output)  // Now q_output_host is a 3D Scala array
        val l = 2
        val n = 0
        val e = 35
        val idx = l * (N * E) + n * E + e
        println("q output value at (l,n,e): " + q_output_host(idx))

        //TO DO: test that the q_output matches the q_proj


        assert(true)
    }
}
