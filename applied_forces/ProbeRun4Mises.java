import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeRun4Mises {
  private static void p(String s) { System.out.println(s); }

  private static void eval(Model m, String tag, String type, String expr) {
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, type);
      m.result().numerical(tag).set("data", "dset6");
      m.result().numerical(tag).set("expr", new String[]{expr});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).setResult();
      double[][] v = m.result().numerical(tag).getReal();
      if (v != null && v.length > 0 && v[0].length > 0) p(tag + "=" + v[0][0]);
      else p(tag + " unavailable");
    } catch (Exception e) {
      p(tag + " failed: " + e.getMessage());
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/FindSolvableMeshSubset_Run4_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    eval(m, "mxv", "MaxVolume", "solid.mises");
    eval(m, "mnv", "MinVolume", "solid.mises");
    eval(m, "mxs", "MaxSurface", "solid.mises");
    eval(m, "mns", "MinSurface", "solid.mises");

    try {
      m.result().numerical().create("mxs_snout", "MaxSurface");
      m.result().numerical("mxs_snout").set("data", "dset6");
      m.result().numerical("mxs_snout").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mxs_snout").selection().named("sel_snout2");
      m.result().numerical("mxs_snout").setResult();
      double[][] v = m.result().numerical("mxs_snout").getReal();
      if (v != null && v.length > 0 && v[0].length > 0) p("mxs_snout=" + v[0][0]);
      else p("mxs_snout unavailable");
    } catch (Exception e) {
      p("mxs_snout failed: " + e.getMessage());
    }
  }
}
