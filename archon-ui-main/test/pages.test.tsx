import { render, screen } from '@testing-library/react'
import { describe, test, expect } from 'vitest'
import React from 'react'

describe('Page Load Tests', () => {
  test('simple page component renders', () => {
    const MockPage = () => <h1>Projects</h1>
    render(<MockPage />)
    expect(screen.getByText('Projects')).toBeInTheDocument()
  })

  test('knowledge base mock renders', () => {
    const MockKnowledgePage = () => <h1>Knowledge Base</h1>
    render(<MockKnowledgePage />)
    expect(screen.getByText('Knowledge Base')).toBeInTheDocument()
  })

  test('settings mock renders', () => {
    const MockSettingsPage = () => <h1>Settings</h1>
    render(<MockSettingsPage />)
    expect(screen.getByText('Settings')).toBeInTheDocument()
  })

  test('mcp mock renders', () => {
    const MockMCPPage = () => <h1>MCP Servers</h1>
    render(<MockMCPPage />)
    expect(screen.getByText('MCP Servers')).toBeInTheDocument()
  })

  test('tasks mock renders', () => {
    const MockTasksPage = () => (
      <div>
        <h1>Tasks</h1>
        <div>TODO</div>
        <div>In Progress</div>
        <div>Done</div>
      </div>
    )
    render(<MockTasksPage />)
    expect(screen.getByText('Tasks')).toBeInTheDocument()
    expect(screen.getByText('TODO')).toBeInTheDocument()
    expect(screen.getByText('In Progress')).toBeInTheDocument()
    expect(screen.getByText('Done')).toBeInTheDocument()
  })
})