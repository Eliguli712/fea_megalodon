import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeRun5Fields {
  private static void eval(Model m, String tag, String type, String expr, String unit) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, type);
      m.result().numerical(tag).set("data", "dset6");
      m.result().numerical(tag).set("expr", new String[]{expr});
      if (unit != null && unit.length() > 0) m.result().numerical(tag).set("unit", new String[]{unit});
      m.result().numerical(tag).setResult();
      double[][] v = m.result().numerical(tag).getReal();
      if (v != null && v.length > 0 && v[0].length > 0) {
        System.out.println(tag + " expr=" + expr + " -> " + v[0][0]);
      } else {
        System.out.println(tag + " expr=" + expr + " unavailable");
      }
    } catch (Exception e) {
      System.out.println(tag + " expr=" + expr + " failed: " + e.getMessage());
    }
  }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/FindSolvableMeshSubset_Run5_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    eval(m, "mx_u", "MaxVolume", "sqrt(u^2+v^2+w^2)", "m");
    eval(m, "mx_sx", "MaxVolume", "solid.sx", "Pa");
    eval(m, "mx_sy", "MaxVolume", "solid.sy", "Pa");
    eval(m, "mx_sz", "MaxVolume", "solid.sz", "Pa");
    eval(m, "mx_mises", "MaxVolume", "solid.mises", "Pa");
    eval(m, "mx_ogpw", "MaxVolume", "solid.hmm_og.pw", "Pa");
  }
}
