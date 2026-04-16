import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class FixAndRunNH {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    m.param().set("muLame", "2.5e7[Pa]");
    m.param().set("lambdaLame", "2.2e8[Pa]");
    m.param().set("youngsmodulus", "1.5e8[Pa]");
    m.param().set("poissonsratio", "0.3");

    try { m.component("comp1").physics("solid").feature("lemm1").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_nh").active(true); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr2").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr5").active(false); } catch (Exception e) {}

    try {
      m.study("std_nh").run();
      System.out.println("std_nh ok");
    } catch (Exception e) {
      System.out.println("std_nh failed: " + e.getMessage());
      e.printStackTrace();
    }

    try { m.result().numerical().remove("mx_nh_fix"); } catch (Exception e) {}
    try {
      m.result().numerical().create("mx_nh_fix", "MaxVolume");
      m.result().numerical("mx_nh_fix").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mx_nh_fix").set("data", "dset1");
      m.result().numerical("mx_nh_fix").setResult();
      double[][] r = m.result().numerical("mx_nh_fix").getReal();
      System.out.println("mx_nh=" + (r!=null&&r.length>0&&r[0].length>0?r[0][0]:Double.NaN));
    } catch (Exception e) {
      System.out.println("eval failed: " + e.getMessage());
    }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/FixAndRunNH_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
