import spatial.dsl._

@spatial class MultiHeadAttention extends SpatialApp {

  val L = 16  // Sequence length
  val N = 2   // Batch size
  val E = 32  // Embedding size
  val nhead = 4
  val head_dim = E / nhead

  // DRAMs for inputs and outputs
  val Q = DRAM[Float](L, N, E)
  val K = DRAM[Float](L, N, E)
  val V = DRAM[Float](L, N, E)
  val Wq = DRAM[Float](E, E)
  val Wk = DRAM[Float](E, E)
  val Wv = DRAM[Float](E, E)
  val Wo = DRAM[Float](E, E)
  val Out = DRAM[Float](L, N, E)

  @struct case class Vector32(data: Array[Float32]) // E = 32

  Accel {
    val q_sram = SRAM[Float](L, N, E)
    val k_sram = SRAM[Float](L, N, E)
    val v_sram = SRAM[Float](L, N, E)
    val wq_sram = SRAM[Float](E, E)
    val wk_sram = SRAM[Float](E, E)
    val wv_sram = SRAM[Float](E, E)
    val wo_sram = SRAM[Float](E, E)
    val out_sram = SRAM[Float](L, N, E)

    // Load all weights and inputs
    q_sram load Q
    k_sram load K
    v_sram load V
    wq_sram load Wq
    wk_sram load Wk
    wv_sram load Wv
    wo_sram load Wo

    val scale = 1.0f / Math.sqrt(head_dim.to[Float])

    // Q_proj = Q x Wq
    val q_proj = SRAM[Float](L, N, E)
    Foreach(L by 1, N by 1, E by 1){ (l,n,e) =>
      q_proj(l,n,e) = Reduce(E by 1)(0f){ i =>
        q_sram(l,n,i) * wq_sram(e, i)
      }{_+_} * scale
    }

    // Same for K_proj, V_proj
    val k_proj = SRAM[Float](L, N, E)
    val v_proj = SRAM[Float](L, N, E)
    Foreach(L by 1, N by 1, E by 1){ (l,n,e) =>
      k_proj(l,n,e) = Reduce(E by 1)(0f){ i => k_sram(l,n,i) * wk_sram(e, i) }{_+_}
      v_proj(l,n,e) = Reduce(E by 1)(0f){ i => v_sram(l,n,i) * wv_sram(e, i) }{_+_}
    }

    // Compute attention scores: q_proj . k_proj^T per head
    val attn_scores = SRAM[Float](N * nhead, L, L)
    Foreach(N by 1, nhead by 1, l1 by 1, l2 by 1){ (n,h,l1,l2) =>
      val offset = h * head_dim
      val dot = Reduce(head_dim by 1)(0f){ i =>
        q_proj(l1,n,offset+i) * k_proj(l2,n,offset+i)
      }{_+_}
      attn_scores(n*nhead + h, l1, l2) = dot
    }

    // Softmax over L2
    val attn_weights = SRAM[Float](N * nhead, L, L)
    Foreach(N * nhead by 1, L by 1){ (h, l1) =>
      val maxval = Reduce(L by 1)(-1e9f){ l2 => attn_scores(h, l1, l2) }{(a,b) => mux(a > b, a, b)}
      val expsum = Reduce(L by 1)(0f){ l2 =>
        val expval = Math.exp(attn_scores(h, l1, l2) - maxval)
        expval
      }{_+_}
      Foreach(L by 1){ l2 =>
        val expval = Math.exp(attn_scores(h, l1, l2) - maxval)
        attn_weights(h, l1, l2) = expval / expsum
      }
    }

    // Weighted sum with V
    val attn_output = SRAM[Float](L, N, E)
    Foreach(N by 1, h by 1, l1 by 1, d by 1){ (n,h,l1,d) =>
      val offset = h * head_dim
      val out = Reduce(L by 1)(0f){ l2 =>
        attn_weights(n*nhead + h, l1, l2) * v_proj(l2,n,offset+d)
      }{_+_}
      attn_output(l1,n,offset+d) = out
    }

    // Final linear projection
    Foreach(L by 1, N by 1, E by 1){ (l,n,e) =>
      out_sram(l,n,e) = Reduce(E by 1)(0f){ i =>
        attn_output(l,n,i) * wo_sram(e, i)
      }{_+_}
    }

    // Store output
    Out store out_sram
  }
}
