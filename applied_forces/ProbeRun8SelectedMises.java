import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.LinkedHashSet;

public class ProbeRun8SelectedMises {
  private static int[] keptDomains() {
    int[] rem = new int[]{2,5,6,25,28,46,48,51,62,84,91,98,100,106,110,116,121,127,131,135,152,165,182};
    LinkedHashSet<Integer> s = new LinkedHashSet<Integer>();
    for (int i=1;i<=183;i++) s.add(i);
    for (int r: rem) s.remove(r);
    int[] out = new int[s.size()];
    int k=0; for (int v: s) out[k++] = v;
    return out;
  }

  private static void eval(Model m, String tag, String type, String expr, int dim) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, type);
      m.result().numerical(tag).set("data", "dset6");
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).selection().geom("geom1", dim);
      m.result().numerical(tag).selection().set(keptDomains());
      m.result().numerical(tag).setResult();
      double[][] v = m.result().numerical(tag).getReal();
      if (v != null && v.length > 0 && v[0].length > 0) {
        System.out.println(tag + "=" + v[0][0]);
      } else {
        System.out.println(tag + " unavailable");
      }
    } catch (Exception e) {
      System.out.println(tag + " failed: " + e.getMessage());
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/FindSolvableMeshSubset_Run8_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    eval(m, "mxu", "MaxVolume", "sqrt(u^2+v^2+w^2)", 3);
    eval(m, "mxm", "MaxVolume", "solid.mises", 3);
  }
}
