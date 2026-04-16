import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeMainDomainSolve {
  private static void p(String s){ System.out.println(s); }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.geom("part1").feature().remove("tor1"); p("removed tor1"); } catch (Exception e) { p("tor1 remove skipped"); }

    int[] mainDom = new int[]{3};

    m.param().set("thrust_load", "5e2[Pa]");

    try {
      m.component("comp1").physics("solid").selection().geom("geom1",3);
      m.component("comp1").physics("solid").selection().set(mainDom);
      p("solid domain set to 3");
    } catch (Exception e) { p("solid selection failed: " + e.getMessage()); }

    try { m.component("comp1").physics("solid").feature("lemm1").selection().geom("geom1",3); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").selection().set(mainDom); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("E_mat","userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("E","1.5e8[Pa]"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("nu_mat","userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("nu","0.3"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("rho_mat","userdef"); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").set("rho","1100[kg/m^3]"); } catch (Exception e) {}

    try { m.component("comp1").selection().remove("sel_tail_fix"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_tail_fix", "Box");
    m.component("comp1").selection("sel_tail_fix").set("entitydim", "2");
    m.component("comp1").selection("sel_tail_fix").set("xmin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("xmax", "1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("ymin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("ymax", "1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("zmin", "-1e9[m]");
    m.component("comp1").selection("sel_tail_fix").set("zmax", "4[m]");

    try {
      int[] b = m.component("comp1").selection("sel_tail_fix").entities(2);
      p("sel_tail_fix count=" + (b==null?-1:b.length));
    } catch (Exception e) { p("sel_tail_fix count read failed"); }

    try { m.component("comp1").selection().remove("sel_snout2"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_snout2", "Box");
    m.component("comp1").selection("sel_snout2").set("entitydim", "2");
    m.component("comp1").selection("sel_snout2").set("xmin", "-1e9[m]");
    m.component("comp1").selection("sel_snout2").set("xmax", "1e9[m]");
    m.component("comp1").selection("sel_snout2").set("ymin", "-1e9[m]");
    m.component("comp1").selection("sel_snout2").set("ymax", "1e9[m]");
    m.component("comp1").selection("sel_snout2").set("zmin", "20[m]");
    m.component("comp1").selection("sel_snout2").set("zmax", "1e9[m]");

    try {
      int[] b = m.component("comp1").selection("sel_snout2").entities(2);
      p("sel_snout2 count=" + (b==null?-1:b.length));
    } catch (Exception e) { p("sel_snout2 count read failed"); }

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

    try {
      m.study("std1").feature("stat").set("mesh", new String[][]{{"geom1","mesh2"}});
      m.study("std1").feature("stat").set("geometricNonlinearity", "off");
      m.study("std1").feature("stat").set("plot", "off");
    } catch (Exception e) { p("study setup failed: " + e.getMessage()); }

    try {
      m.study("std1").run();
      p("std1 run complete");
    } catch (Exception e) {
      p("std1 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try { m.result().numerical().remove("mx_main"); } catch (Exception e) {}
    try {
      m.result().numerical().create("mx_main", "MaxVolume");
      m.result().numerical("mx_main").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mx_main").set("unit", new String[]{"Pa"});
      m.result().numerical("mx_main").set("data", "dset6");
      m.result().numerical("mx_main").selection().geom("geom1",3);
      m.result().numerical("mx_main").selection().set(mainDom);
      m.result().numerical("mx_main").setResult();
      double[][] r = m.result().numerical("mx_main").getReal();
      p("max main-domain mises=" + (r!=null && r.length>0 && r[0].length>0 ? r[0][0] : Double.NaN));
    } catch (Exception e) {
      p("mx_main eval failed: " + e.getMessage());
    }

    try {
      try { m.result().remove("pg_vms_main"); } catch (Exception e) {}
      m.result().create("pg_vms_main", "PlotGroup3D");
      m.result("pg_vms_main").set("data", "dset6");
      m.result("pg_vms_main").create("surf1", "Surface");
      m.result("pg_vms_main").feature("surf1").set("expr", "solid.mises");
      m.result("pg_vms_main").run();
      p("pg_vms_main built");
    } catch (Exception e) { p("plot failed: " + e.getMessage()); }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeMainDomainSolve_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
