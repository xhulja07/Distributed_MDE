import java.lang.{ Integer => JavaInt, Long => JavaLong, String => JavaString, Double => JavaFloat }
import scala.io.Source
import java.util
import java.util.Map.Entry
import java.util.Timer
import javax.cache.processor.{ EntryProcessor, MutableEntry }
import org.apache.ignite.cache.query.SqlFieldsQuery
import org.apache.ignite.internal.util.scala.impl
import org.apache.ignite.scalar.scalar
import org.apache.ignite.scalar.scalar._
import org.apache.ignite.stream.StreamReceiver
import org.apache.ignite.{ IgniteCache, IgniteException }
import scala.util.Random
import scala.io.Source._
import scala.collection.JavaConverters._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import java.util.Calendar;
import java.io.BufferedWriter
import java.io._
import scala.collection.mutable.ListBuffer
import org.apache.commons.math3.analysis.function.Sqrt
import scala.util.control.Breaks._
import java.util.Collection
import org.apache.ignite.lang.IgniteCallable
import org.apache.ignite.lang.IgniteReducer
import java.util.ArrayList
import org.apache.ignite.Ignite
import org.apache.ignite.Ignition
import org.apache.ignite.compute.{ ComputeJob, ComputeJobResult, ComputeTaskSplitAdapter }
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map
import java.nio.file.Paths
import org.apache.ignite.IgniteSystemProperties
import jdk.nashorn.internal.ir.ForNode
import java.util.UUID
import org.apache.ignite.messaging.MessagingListenActor
import scala.collection.JavaConversions._

class ComputeInsideNode(entityCache: Map[Int, DenseVector[Double]], relationCache: Map[Int, DenseVector[Double]], variables: Map[String, String]) {

  var dimNum = Integer.parseInt(variables("dim_num"))
  var batchsize = Integer.parseInt(variables("batchsize"))
  var margin = variables("margin").toDouble
  var rate = variables("rate").toDouble
  var batchesPerNode = Integer.parseInt(variables("batchesPerNode"))

  val L1_flag = true
  /** Type alias. */
  type Cache = IgniteCache[JavaString, JavaInt]
  type CacheFilter = IgniteCache[TripleID, Int]
  type TriplesCache = IgniteCache[Int, TripleID]

  def rand_max(x: Int): Int = {
    var res = (scala.util.Random.nextInt() * scala.util.Random.nextInt()) % x
    while (res < 0)
      res += x
    return res
  }

  def sqr(x: Double): Double = {
    return x * x
  }

  def calc_sum(triple: TripleID): Double = {
    var sum: Double = 0
    var head = entityCache(triple.head)
    var tail = entityCache(triple.tail)
    var rel = relationCache(triple.rel)
    if (L1_flag)
      for (i <- 0 until dimNum)
        sum += math.abs(tail(i) - head(i) - rel(i))
    else
      for (i <- 0 until dimNum)
        sum += sqr(tail(i) - head(i) - rel(i))

    return sum
  }

  def calc_score4DistMult(triple: TripleID): Double = {
    var res: Double = 0
    var head = entityCache(triple.head)
    var tail = entityCache(triple.tail)
    var rel = relationCache(triple.rel)
  
    for (i <- 0 until dimNum) {
      res += (head(i) *  tail(i) * rel(i))
    }

    return res
  }

  def regul_func(triple: TripleID): Double = {
    var sum: Double = 0
    var head = entityCache(triple.head)
    var tail = entityCache(triple.tail)
    var rel = relationCache(triple.rel)
    for (i <- 0 until dimNum) {
      sum += (head(i) * head(i) + tail(i) * tail(i) + rel(i) * rel(i)) / 3
    }

    sum
  }

  def sign(nr: Double): Double = {
    var r: Double = 0
    if (nr > 0)
      r = 1
    else
      r = -1
    r
  }

  def vec_length(a: DenseVector[Double]): Double = {
    var res: Double = 0
    for (i <- 0 until a.size)
      res += a(i) * a(i)
    res = sqr(res)
    return res
  }

  def gradient(trueTriple: TripleID, falseTriple: TripleID, entity_tmp: Map[Int, DenseVector[Double]], relation_tmp: Map[Int, DenseVector[Double]]) {
    var head = entityCache(trueTriple.head)
    var tail = entityCache(trueTriple.tail)
    var rel = relationCache(trueTriple.rel)
    var headF = entityCache(falseTriple.head)
    var tailF = entityCache(falseTriple.tail)
    var relF = relationCache(falseTriple.rel)
    var gradTrue: DenseVector[Double] = DenseVector.zeros[Double](dimNum)
    for (i <- 0 until dimNum)
      gradTrue(i) = (tail(i) - head(i) - rel(i)) * 2.0
    if (L1_flag)
      gradTrue.map(i => { sign(i) })

    if (!entity_tmp.contains(trueTriple.head))
      entity_tmp.put(trueTriple.head, DenseVector.zeros[Double](dimNum))
    if (!entity_tmp.contains(trueTriple.tail))
      entity_tmp.put(trueTriple.tail, DenseVector.zeros[Double](dimNum))
    if (!relation_tmp.contains(trueTriple.rel))
      relation_tmp.put(trueTriple.rel, DenseVector.zeros[Double](dimNum))

    if (trueTriple.head == falseTriple.head) {
      if (!entity_tmp.contains(falseTriple.tail))
        entity_tmp.put(falseTriple.tail, DenseVector.zeros[Double](dimNum))
    } else {
      if (!entity_tmp.contains(falseTriple.head))
        entity_tmp.put(falseTriple.head, DenseVector.zeros[Double](dimNum))
    }

    var h = entity_tmp(trueTriple.head) + gradTrue * rate
    entity_tmp.put(trueTriple.head, h)

    var t = entity_tmp(trueTriple.tail) - gradTrue * rate
    entity_tmp.put(trueTriple.tail, t)

    var r = relation_tmp(trueTriple.rel) + gradTrue * rate
    relation_tmp.put(trueTriple.rel, r)

    var gradFalse: DenseVector[Double] = (tailF - headF - relF) * 2.0
    if (L1_flag)
      gradFalse.map(i => { sign(i) })

    var hF = entity_tmp(falseTriple.head) - gradFalse * rate
    entity_tmp.put(falseTriple.head, hF)

    var tF = entity_tmp(falseTriple.tail) + gradFalse * rate
    entity_tmp.put(falseTriple.tail, tF)

    var rF = relation_tmp(falseTriple.rel) - gradFalse * rate
    relation_tmp.put(falseTriple.rel, rF)
  }

  def createRelationMatrix(rel: DenseVector[Double]): DenseMatrix[Double] = {

    val relDenseMatrix = DenseMatrix.zeros[Double](dimNum, dimNum)
    for (i <- 0 until dimNum)
      relDenseMatrix(i, i) = rel(i)

    relDenseMatrix
  }

  def gradient4DistMult(trueTriple: TripleID, falseTriple: TripleID, entity_tmp: Map[Int, DenseVector[Double]], relation_tmp: Map[Int, DenseVector[Double]]) {
    var head = entityCache(trueTriple.head)
    var tail = entityCache(trueTriple.tail)
    var rel = relationCache(trueTriple.rel)

    var headF = entityCache(falseTriple.head)
    var tailF = entityCache(falseTriple.tail)
    var relF = relationCache(falseTriple.rel)

    //update the temporary embeddings maps
    if (!entity_tmp.contains(trueTriple.head))
      entity_tmp.put(trueTriple.head, DenseVector.zeros[Double](dimNum))
    if (!entity_tmp.contains(trueTriple.tail))
      entity_tmp.put(trueTriple.tail, DenseVector.zeros[Double](dimNum))
    if (!relation_tmp.contains(trueTriple.rel))
      relation_tmp.put(trueTriple.rel, DenseVector.zeros[Double](dimNum))

    if (trueTriple.head == falseTriple.head) {
      if (!entity_tmp.contains(falseTriple.tail))
        entity_tmp.put(falseTriple.tail, DenseVector.zeros[Double](dimNum))
    } else {
      if (!entity_tmp.contains(falseTriple.head))
        entity_tmp.put(falseTriple.head, DenseVector.zeros[Double](dimNum))
    }

    var gradTrueH: DenseVector[Double] = DenseVector.zeros[Double](dimNum)
    var gradTrueT: DenseVector[Double] = DenseVector.zeros[Double](dimNum)
    var gradTrueR: DenseVector[Double] = DenseVector.zeros[Double](dimNum)

    for (i <- 0 until dimNum) {
      gradTrueH(i) = tail(i) * rel(i)
      gradTrueT(i) = head(i) * rel(i)
      gradTrueR(i) = tail(i) * head(i)

      //      gradTrueH(i) = (head(i) * tail(i) * rel(i)) * tail(i) * rel(i) * 2
      //      gradTrueT(i) = (head(i) * tail(i) * rel(i)) * head(i) * rel(i) * 2
      //      gradTrueR(i) = (head(i) * tail(i) * rel(i)) * tail(i) * head(i) * 2
    }
    var regulator: Double = regul_func(trueTriple) * 0.005

    var gradT: DenseVector[Double] = gradTrueH + gradTrueT + gradTrueR
    gradT = gradT * rate

    var h = entity_tmp(trueTriple.head) + gradTrueH * rate 
    entity_tmp.put(trueTriple.head, h)

    var t = entity_tmp(trueTriple.tail) - gradTrueT * rate 
    entity_tmp.put(trueTriple.tail, t)

    var r = relation_tmp(trueTriple.rel) + gradTrueR * rate 
    relation_tmp.put(trueTriple.rel, r)

    var gradFalseH: DenseVector[Double] = DenseVector.zeros[Double](dimNum)
    var gradFalseT: DenseVector[Double] = DenseVector.zeros[Double](dimNum)
    var gradFalseR: DenseVector[Double] = DenseVector.zeros[Double](dimNum)

    for (i <- 0 until dimNum) {
      //      gradFalseH(i) = (headF(i) * tailF(i) * relF(i)) * tailF(i) * relF(i) * 2
      //      gradFalseT(i) = (headF(i) * tailF(i) * relF(i)) * headF(i) * relF(i) * 2
      //      gradFalseR(i) = (headF(i) * tailF(i) * relF(i)) * tailF(i) * headF(i) * 2
      gradFalseH(i) = tailF(i) * relF(i)
      gradFalseT(i) = headF(i) * relF(i)
      gradFalseR(i) = tailF(i) * headF(i)
    }

    var regulatorF: Double = regul_func(falseTriple) * 0.005
    var gradF: DenseVector[Double] = gradFalseH + gradFalseT + gradFalseR
    gradT = gradT * rate

    var hF = entity_tmp(falseTriple.head) - gradFalseH * rate
    entity_tmp.put(falseTriple.head, hF)

    var tF = entity_tmp(falseTriple.tail) + gradFalseT * rate 
    entity_tmp.put(falseTriple.tail, tF)

    var rF = relation_tmp(falseTriple.rel) - gradFalseR * rate
    relation_tmp.put(falseTriple.rel, rF)
  }

  def train_kb(trueTriple: TripleID, falseTriple: TripleID, entity_tmp: Map[Int, DenseVector[Double]], relation_tmp: Map[Int, DenseVector[Double]]): Double = {
    var res: Double = 0
    var trueDist = calc_score4DistMult(trueTriple)
    var FalseDist = calc_score4DistMult(falseTriple)
    if (trueDist + margin > FalseDist) {
      res += margin + trueDist - FalseDist
      //      gradient(trueTriple, falseTriple, entity_tmp, relation_tmp)
      gradient4DistMult(trueTriple, falseTriple, entity_tmp, relation_tmp)

    }

    res
  }

  def computeInsideNode(): Tuple3[Double, Map[Int, DenseVector[Double]], Map[Int, DenseVector[Double]]] = {
    val remoteIgnite = Ignition.localIgnite()
    var trainingTriplesCache: IgniteCache[JavaInt, TripleID] = remoteIgnite.cache("tripleIde");
    var existingTriplesCache: IgniteCache[TripleID, JavaInt] = remoteIgnite.cache("allTriplesCache");
    var entity_tmp: Map[Int, DenseVector[Double]] = Map.empty[Int, DenseVector[Double]]
    var relation_tmp: Map[Int, DenseVector[Double]] = Map.empty[Int, DenseVector[Double]]
    var entity_num = entityCache.size
    var relation_num = relationCache.size
    var triple_num = trainingTriplesCache.size()

    var totRes: Double = 0
    for (batch <- 0 until batchesPerNode) {
      for (i <- 0 until batchsize) {
        var id = rand_max(triple_num)
        var randomEntity = rand_max(entity_num)
        var triple: TripleID = trainingTriplesCache.get(id)

        if (scala.util.Random.nextInt() % 1000 < 500) {
          while (randomEntity == triple.tail || existingTriplesCache.get(new TripleID(triple.head, randomEntity, triple.rel)) != null) {
            randomEntity = rand_max(entity_num)
          }
          if (!entity_tmp.contains(randomEntity))
            totRes += train_kb(triple, new TripleID(triple.head, randomEntity, triple.rel), entity_tmp, relation_tmp)
        } else {
          while (randomEntity == triple.head || existingTriplesCache.get(new TripleID(randomEntity, triple.tail, triple.rel)) != null) {
            randomEntity = rand_max(entity_num)
          }
          if (!entity_tmp.contains(randomEntity))
            totRes += train_kb(triple, new TripleID(randomEntity, triple.tail, triple.rel), entity_tmp, relation_tmp)
        }
      }
    }
    (totRes, entity_tmp, relation_tmp)
  }

}