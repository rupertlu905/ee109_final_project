import spatial.dsl._

@spatial class MultiheadAttention extends SpatialTest {
  type T = Float

  def main(args: Array[String]): Unit = {
    val q = loadCSV2D[T]("data_dump/model.transformer.decoder.layers.0.multihead_attn/q.csv", ",", "\n")
    val k = loadCSV2D[T]("data_dump/model.transformer.decoder.layers.0.multihead_attn/k.csv", ",", "\n")
    val v = loadCSV2D[T]("data_dump/model.transformer.decoder.layers.0.multihead_attn/v.csv", ",", "\n")

    Accel {

    }

    printMatrix(q, "q")
    printMatrix(k, "k")
    printMatrix(v, "v")

    assert(true)
  }
}