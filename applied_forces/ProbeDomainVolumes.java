import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeDomainVolumes {
  private static int[] keptDomains() {
    int[] rem = new int[]{2,5,6,25,28,46,48,51,62,84,91,98,100,106,110,116,121,127,131,135,152,165,182};
    boolean[] keep = new boolean[184];
    for (int i=1;i<=183;i++) keep[i] = true;
    for (int r: rem) keep[r] = false;
    int n = 0;
    for (int i=1;i<=183;i++) if (keep[i]) n++;
    int[] out = new int[n];
    int k = 0;
    for (int i=1;i<=183;i++) if (keep[i]) out[k++] = i;
    return out;
  }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    int[] doms = keptDomains();
    double[] vol = new double[doms.length];

    for (int i=0; i<doms.length; i++) {
      int d = doms[i];
      String tag = "iv" + d;
      try { m.result().numerical().remove(tag); } catch (Exception e) {}
      try {
        m.result().numerical().create(tag, "IntVolume");
        m.result().numerical(tag).set("expr", new String[]{"1"});
        m.result().numerical(tag).set("data", "dset6");
        m.result().numerical(tag).selection().geom("geom1", 3);
        m.result().numerical(tag).selection().set(new int[]{d});
        m.result().numerical(tag).setResult();
        double[][] r = m.result().numerical(tag).getReal();
        vol[i] = (r != null && r.length>0 && r[0].length>0) ? r[0][0] : Double.NaN;
      } catch (Exception e) {
        vol[i] = Double.NaN;
      }
    }

    // print top 30 by volume (selection sort style)
    System.out.println("top domains by volume integral(1):");
    boolean[] used = new boolean[doms.length];
    for (int k=0; k<30 && k<doms.length; k++) {
      int best = -1;
      double bestv = -1e300;
      for (int i=0; i<doms.length; i++) {
        if (used[i]) continue;
        double v = vol[i];
        if (Double.isNaN(v)) continue;
        if (best < 0 || v > bestv) { best = i; bestv = v; }
      }
      if (best < 0) break;
      used[best] = true;
      System.out.println(doms[best] + " " + vol[best]);
    }
  }
}
