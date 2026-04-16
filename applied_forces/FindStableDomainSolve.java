import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class FindStableDomainSolve {
  private static double evalMises(Model m, int dom) {
    String tag = "mxd" + dom;
    try { m.result().numerical().remove(tag); } catch (Exception e) {}
    try {
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).set("data", "dset6");
      m.result().numerical(tag).selection().geom("geom1",3);
      m.result().numerical(tag).selection().set(new int[]{dom});
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      return (r!=null && r.length>0 && r[0].length>0) ? r[0][0] : Double.NaN;
    } catch (Exception e) {
      return Double.NaN;
    }
  }

  private static int[] selCount(Model m, String tag) {
    try {
      int[] b = m.component("comp1").selection(tag).entities(2);
      return b;
    } catch (Exception e) { return null; }
  }

  private static void setupCommon(Model m) {
    try { m.geom("part1").feature().remove("tor1"); } catch (Exception e) {}

    try { m.component("comp1").selection().remove("sel_tail_fix"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_tail_fix", "Box");
    m.component("comp1").selection("sel_tail_fix").set("entitydim", "2");
    m.component("comp1").selection("sel_tail_fix").set("xmin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("xmax", "1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("ymin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("ymax", "1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("zmin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("zmax", "4[m]");

    try { m.component("comp1").selection().remove("sel_snout2"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_snout2", "Box");
    m.component("comp1").selection("sel_snout2").set("entitydim", "2");
    m.component("comp1").selection("sel_snout2").set("xmin", "-1e9[m]");
    m.component("comp1").selection("sel_snout2").set("xmax", "1e9[m]");
    m.component("comp1").selection("sel_snout2").set("ymin", "-1e9[m]");
    m.component("comp1").selection("sel_snout2").set("ymax", "1e9[m]");
    m.component("comp1").selection("sel_snout2").set("zmin", "20[m]");
    m.component("comp1").selection("sel_snout2").set("zmax", "1e9[m]");

    try { m.component("comp1").physics("solid").feature("fix1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("fix1", "Fixed", 2); }
    m.component("comp1").physics("solid").feature("fix1").selection().named("sel_tail_fix");
    m.component("comp1").physics("solid").feature("fix1").active(true);

    try { m.component("comp1").physics("solid").feature("rms1").active(false); } catch (Exception e) {}

    try { m.component("comp1").physics("solid").feature("bndl1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("bndl1", "BoundaryLoad", 2); }
    m.component("comp1").physics("solid").feature("bndl1").selection().named("sel_snout2");
    m.component("comp1").physics("solid").feature("bndl1").set("forceType", "ForceArea");
    m.component("comp1").physics("solid").feature("bndl1").set("force_src", "userdef");
    m.component("comp1").physics("solid").feature("bndl1").set("force", new String[]{"0","0","thrust_load"});
    try { m.component("comp1").physics("solid").feature("bndl1").set("forceReferenceArea_src", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl1").set("forceReferenceArea", new String[]{"0","0","thrust_load"}); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl1").set("forceDeformedArea_src", "userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl1").set("forceDeformedArea", new String[]{"0","0","thrust_load"}); } catch (Exception e) {}
    m.component("comp1").physics("solid").feature("bndl1").active(true);
    try { m.component("comp1").physics("solid").feature("bndl_pr").active(false); } catch (Exception e) {}

    try { m.component("comp1").physics("solid").feature("hmm_nh").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr2").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr5").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").active(true); } catch (Exception e) {}

    m.param().set("thrust_load", "50[Pa]");

    try {
      m.study("std1").feature("stat").set("mesh", new String[][]{{"geom1","mesh2"}});
      m.study("std1").feature("stat").set("geometricNonlinearity", "off");
      m.study("std1").feature("stat").set("plot", "off");
    } catch (Exception e) {}
  }

  public static void main(String[] args) {
    int[] cand = new int[]{3,126,26,130,109,44,1,52,150,111,117,112,138,181,4,83,85,58,149,69};

    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    setupCommon(m);

    int[] tail = selCount(m, "sel_tail_fix");
    int[] snout = selCount(m, "sel_snout2");
    System.out.println("tail count=" + (tail==null?-1:tail.length) + " snout count=" + (snout==null?-1:snout.length));

    for (int d : cand) {
      try {
        m.component("comp1").physics("solid").selection().geom("geom1",3);
        m.component("comp1").physics("solid").selection().set(new int[]{d});
        m.component("comp1").physics("solid").feature("lemm1").selection().geom("geom1",3);
        m.component("comp1").physics("solid").feature("lemm1").selection().set(new int[]{d});
        m.component("comp1").physics("solid").feature("lemm1").set("E_mat","userdef");
        m.component("comp1").physics("solid").feature("lemm1").set("E","1.5e8[Pa]");
        m.component("comp1").physics("solid").feature("lemm1").set("nu_mat","userdef");
        m.component("comp1").physics("solid").feature("lemm1").set("nu","0.3");
        m.component("comp1").physics("solid").feature("lemm1").set("rho_mat","userdef");
        m.component("comp1").physics("solid").feature("lemm1").set("rho","1100[kg/m^3]");
      } catch (Exception e) {}

      boolean ok = true;
      try { m.study("std1").run(); }
      catch (Exception e) { ok = false; }

      double mx = evalMises(m, d);
      System.out.println("domain=" + d + " ok=" + ok + " max_mises=" + mx);
    }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/FindStableDomainSolve_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
