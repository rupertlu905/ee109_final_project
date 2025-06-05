import scala.io.Source
import spatial.dsl._

//takes in CSV, NO COMMA SEPARATION, separate line for each element
//total # of lines = total # of elements

@spatial class MultiheadAttention2 extends SpatialTest {
    // TODO: tweak the precision
    type T = FixPt[TRUE,_32,_24] // Fixed point notation, [signed, 16 bits, 8 bits fraction]


    override def runtimeArgs = "0"
    def main(args: Array[String]): Unit = {
        val N = 1
        val L_q = 2
        val L_k = 3
        val E = 6
        val n_heads = 2

        val query_values = loadCSV1D[T]("data_dump/test2/query.csv", "\n")
        val key_values = loadCSV1D[T]("data_dump/test2/key.csv", "\n")
        val gold_attn_scores = loadCSV1D[T]("data_dump/test2/attn_scores.csv", "\n")
        val gold_attn_weights = loadCSV1D[T]("data_dump/test2/attn_weights.csv", "\n")

        // DRAMs for inputs
        val query_dram = DRAM[T](L_q, N, E)
        val key_dram = DRAM[T](L_k, N, E)

        // DRAM for output
        val output_dram = DRAM[T](N*n_heads, L_q, L_k)

        //setMem step 
        setMem(query_dram, query_values)
        setMem(key_dram, key_values)

        
        Accel {
            val query_sram = SRAM[T](L_q, N, E)
            val key_sram = SRAM[T](L_k, N, E)
            val output_sram = SRAM[T](N*n_heads, L_q, L_k)

             // Load input matrices into SRAM
            query_sram load query_dram
            key_sram load key_dram

            Foreach(N by 1, n_heads by 1, L_q by 1) { (n, h, l_q) =>
                val head_dim = E / n_heads
                val attn_scores = SRAM[T](L_k)
                Foreach(L_k by 1) { l_k =>
                    val sum = Reg[T](0.to[T])
                    Foreach(head_dim by 1) { d =>
                        val q_idx = l_q * E + h * head_dim + d
                        val k_idx = l_k * E + h * head_dim + d
                        sum := sum + query_sram(l_q, n, h * head_dim + d) * key_sram(l_k, n, h * head_dim + d)
                    }
                    // output_sram(n * n_heads + h, l_q, l_k) = sum
                    attn_scores(l_k) = sum
                }
                val exp_sum = Reg[T](0.to[T])
                Foreach(L_k by 1) { l_k =>
                    exp_sum := exp_sum + exp(attn_scores(l_k))
                }
                Foreach(L_k by 1) { l_k =>
                    output_sram(n * n_heads + h, l_q, l_k) = exp(attn_scores(l_k)) / exp_sum
                }
            }

            output_dram store output_sram
        }

        //test that q_output is created at all; print one specific element, see if any output is created
        val accel_attn_weights = getMem(output_dram)  // Now q_output_host is a 3D Scala array
        printArray(accel_attn_weights, "accel_attn_weights")
        printArray(gold_attn_weights, "gold_attn_weights")

        //TODO: test that the accel_attn_weights matches the gold_attn_weights
        val cksum = accel_attn_weights.zip(gold_attn_weights){_==_}.reduce{_&&_}
        assert(cksum)
    }
}
