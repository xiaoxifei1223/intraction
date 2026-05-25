import { Routes, Route } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { RouteGuard } from './RouteGuard';

// Shared pages
import { LandingPage } from '@/features/shared/pages/LandingPage';
import { SkillDetailPage } from '@/features/shared/pages/SkillDetailPage';
import { IncidentDetailPage } from '@/features/shared/pages/IncidentDetailPage';
import { SearchResultPage } from '@/features/shared/pages/SearchResultPage';
import { RoleMismatchPage } from '@/features/shared/pages/RoleMismatchPage';
import { TeamsSimulatorPage } from '@/features/shared/pages/TeamsSimulatorPage';
import { VSCodeSimulatorPage } from '@/features/shared/pages/VSCodeSimulatorPage';

// Layer 1
import { DiagnosePage } from '@/features/layer1/pages/DiagnosePage';
import { MySkillsPage } from '@/features/layer1/pages/MySkillsPage';
import { SnippetVaultPage } from '@/features/layer1/pages/SnippetVaultPage';
import { LearningMapPage } from '@/features/layer1/pages/LearningMapPage';
import { ArenaPage } from '@/features/layer1/pages/ArenaPage';
import { ProfilePage } from '@/features/layer1/pages/ProfilePage';

// Layer 2
import { TeamOverviewPage } from '@/features/layer2/pages/TeamOverviewPage';
import { SkillRadarPage } from '@/features/layer2/pages/SkillRadarPage';
import { MTTRAnalysisPage } from '@/features/layer2/pages/MTTRAnalysisPage';
import { MembersPage } from '@/features/layer2/pages/MembersPage';
import { SchedulingPage } from '@/features/layer2/pages/SchedulingPage';
import { ReportsPage } from '@/features/layer2/pages/ReportsPage';

// Layer 3
import { ExecutiveDashboardPage } from '@/features/layer3/pages/ExecutiveDashboardPage';
import { AIGovernancePage } from '@/features/layer3/pages/AIGovernancePage';
import { StrategyAlignPage } from '@/features/layer3/pages/StrategyAlignPage';
import { OrgPlannerPage } from '@/features/layer3/pages/OrgPlannerPage';
import { MaturityAssessmentPage } from '@/features/layer3/pages/MaturityAssessmentPage';
import { BoardReportPage } from '@/features/layer3/pages/BoardReportPage';

interface AppRoute {
  path: string;
  element: React.ReactNode;
  roles: string[];
}

const routes: AppRoute[] = [
  { path: '/', element: <LandingPage />, roles: ['all'] },
  { path: '/diagnose', element: <DiagnosePage />, roles: ['engineer', 'lead', 'executive'] },
  { path: '/my-skills', element: <MySkillsPage />, roles: ['engineer', 'lead'] },
  { path: '/snippets', element: <SnippetVaultPage />, roles: ['engineer', 'lead'] },
  { path: '/learning', element: <LearningMapPage />, roles: ['engineer', 'lead'] },
  { path: '/arena', element: <ArenaPage />, roles: ['engineer', 'lead'] },
  { path: '/profile', element: <ProfilePage />, roles: ['engineer', 'lead', 'executive'] },
  { path: '/team', element: <TeamOverviewPage />, roles: ['lead', 'executive'] },
  { path: '/team/radar', element: <SkillRadarPage />, roles: ['lead', 'executive'] },
  { path: '/team/mttr', element: <MTTRAnalysisPage />, roles: ['lead', 'executive'] },
  { path: '/team/members', element: <MembersPage />, roles: ['lead', 'executive'] },
  { path: '/team/schedule', element: <SchedulingPage />, roles: ['lead'] },
  { path: '/team/reports', element: <ReportsPage />, roles: ['lead', 'executive'] },
  { path: '/executive', element: <ExecutiveDashboardPage />, roles: ['executive'] },
  { path: '/executive/governance', element: <AIGovernancePage />, roles: ['executive'] },
  { path: '/executive/strategy', element: <StrategyAlignPage />, roles: ['executive'] },
  { path: '/executive/planner', element: <OrgPlannerPage />, roles: ['executive'] },
  { path: '/executive/maturity', element: <MaturityAssessmentPage />, roles: ['executive'] },
  { path: '/executive/board-report', element: <BoardReportPage />, roles: ['executive'] },
  { path: '/skill/:skillId', element: <SkillDetailPage />, roles: ['all'] },
  { path: '/incident/:incidentId', element: <IncidentDetailPage />, roles: ['all'] },
  { path: '/search', element: <SearchResultPage />, roles: ['all'] },
  { path: '/simulator/teams', element: <TeamsSimulatorPage />, roles: ['all'] },
  { path: '/simulator/vscode', element: <VSCodeSimulatorPage />, roles: ['all'] },
];

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        {routes.map((route) => (
          <Route
            key={route.path}
            path={route.path}
            element={<RouteGuard allowedRoles={route.roles}>{route.element}</RouteGuard>}
          />
        ))}
        <Route path="/unauthorized" element={<RoleMismatchPage />} />
      </Route>
    </Routes>
  );
}
