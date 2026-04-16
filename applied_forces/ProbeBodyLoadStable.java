import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeBodyLoadStable {
  private static int[] keptDomains() {
    int[] rem = new int[]{2,5,6,25,28,46,48,51,62,84,91,98,100,106,110,116,121,127,131,135,152,165,182};
    boolean[] keep = new boolean[184];
    for (int i=1;i<=183;i++) keep[i] = true;
    for (int r: rem) keep[r] = false;
    int n=0; for (int i=1;i<=183;i++) if (keep[i]) n++;
    int[] out = new int[n]; int k=0;
    for (int i=1;i<=183;i++) if (keep[i]) out[k++]=i;
    return out;
  }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.geom("part1").feature().remove("tor1"); } catch (Exception e) {}

    int[] dom = keptDomains();
    m.param().set("thrust_vol", "2e4[N/m^3]");

    m.component("comp1").physics("solid").selection().geom("geom1",3);
    m.component("comp1").physics("solid").selection().set(dom);

    try { m.component("comp1").selection().remove("sel_snout_dom"); } catch (Exception e) {}
    m.component("comp1").selection().create("sel_snout_dom", "Box");
    m.component("comp1").selection("sel_snout_dom").set("entitydim", "3");
    m.component("comp1").selection("sel_snout_dom").set("xmin", "-1e9[m]");
    m.component("comp1").selection("sel_snout_dom").set("xmax", "1e9[m]");
    m.component("comp1").selection("sel_snout_dom").set("ymin", "-1e9[m]");
    m.component("comp1").selection("sel_snout_dom").set("ymax", "1e9[m]");
    m.component("comp1").selection("sel_snout_dom").set("zmin", "20[m]");
    m.component("comp1").selection("sel_snout_dom").set("zmax", "1e9[m]");

    try {
      int[] sd = m.component("comp1").selection("sel_snout_dom").entities(3);
      System.out.println("sel_snout_dom count=" + (sd==null?-1:sd.length));
    } catch (Exception e) { System.out.println("sel_snout_dom read failed"); }

    try { m.component("comp1").physics("solid").feature("fix1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("fix1", "Fixed", 2); }
    m.component("comp1").physics("solid").feature("fix1").selection().geom("geom1",2);
    m.component("comp1").physics("solid").feature("fix1").selection().all();
    m.component("comp1").physics("solid").feature("fix1").active(true);

    try { m.component("comp1").physics("solid").feature("rms1").active(false); } catch (Exception e) {}

    try { m.component("comp1").physics("solid").feature("bndl1").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("bndl_pr").active(false); } catch (Exception e) {}

    try { m.component("comp1").physics("solid").feature().remove("body1"); } catch (Exception e) {}
    m.component("comp1").physics("solid").create("body1", "BodyLoad", 3);
    m.component("comp1").physics("solid").feature("body1").selection().named("sel_snout_dom");
    m.component("comp1").physics("solid").feature("body1").set("FperVol", new String[]{"0","0","thrust_vol"});
    m.component("comp1").physics("solid").feature("body1").active(true);

    try { m.component("comp1").physics("solid").feature("hmm_nh").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_og").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr2").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("hmm_mr5").active(false); } catch (Exception e) {}
    try { m.component("comp1").physics("solid").feature("lemm1").active(true); } catch (Exception e) {}

    try {
      m.component("comp1").physics("solid").feature("lemm1").selection().geom("geom1",3);
      m.component("comp1").physics("solid").feature("lemm1").selection().set(dom);
      m.component("comp1").physics("solid").feature("lemm1").set("E_mat","userdef");
      m.component("comp1").physics("solid").feature("lemm1").set("E","1.5e8[Pa]");
      m.component("comp1").physics("solid").feature("lemm1").set("nu_mat","userdef");
      m.component("comp1").physics("solid").feature("lemm1").set("nu","0.3");
      m.component("comp1").physics("solid").feature("lemm1").set("rho_mat","userdef");
      m.component("comp1").physics("solid").feature("lemm1").set("rho","1100[kg/m^3]");
    } catch (Exception e) {}

    try { m.study("std1").feature("stat").set("mesh", new String[][]{{"geom1","mesh2"}}); } catch (Exception e) {}
    try { m.study("std1").feature("stat").set("geometricNonlinearity", "off"); } catch (Exception e) {}
    try { m.study("std1").feature("stat").set("plot", "off"); } catch (Exception e) {}

    try {
      m.study("std1").run();
      System.out.println("std1 run complete");
    } catch (Exception e) {
      System.out.println("std1 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try { m.result().numerical().remove("mx_body"); } catch (Exception e) {}
    try {
      m.result().numerical().create("mx_body", "MaxVolume");
      m.result().numerical("mx_body").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mx_body").set("unit", new String[]{"Pa"});
      m.result().numerical("mx_body").set("data", "dset6");
      m.result().numerical("mx_body").selection().geom("geom1",3);
      m.result().numerical("mx_body").selection().set(dom);
      m.result().numerical("mx_body").setResult();
      double[][] r = m.result().numerical("mx_body").getReal();
      System.out.println("max mises=" + (r!=null&&r.length>0&&r[0].length>0 ? r[0][0] : Double.NaN));
    } catch (Exception e) {
      System.out.println("eval failed: " + e.getMessage());
    }

    try {
      try { m.result().remove("pg_body"); } catch (Exception e) {}
      m.result().create("pg_body", "PlotGroup3D");
      m.result("pg_body").set("data", "dset6");
      m.result("pg_body").create("surf1", "Surface");
      m.result("pg_body").feature("surf1").set("expr", "solid.mises");
      m.result("pg_body").run();
      System.out.println("pg_body built");
    } catch (Exception e) {
      System.out.println("plot failed: " + e.getMessage());
    }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeBodyLoadStable_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}
